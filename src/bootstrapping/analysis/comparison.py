"""Functions for comparing model predictions against the gold standard."""

from __future__ import annotations

from typing import Any

import conllu
import numpy as np
import pandas as pd
from conllu_tools.utils import feature_dict_to_string
from scipy import stats

from bootstrapping.extractors import extract_unique_lemmata, extract_upos_counts
from bootstrapping.parsers import get_token_language


def _character_spans(sentence: conllu.TokenList) -> list[tuple[conllu.Token, int, int]]:
    """Return (token, start, end) for each word."""
    spans = []
    offset = 0
    for token in sentence:
        # multiword range rows are re-materialised by their component words
        # so counting both would double the stream.
        if not isinstance(token['id'], int):
            continue
        form = ''.join((token['form'] or '').split())
        spans.append((token, offset, offset + len(form)))
        offset += len(form)
    return spans


def align_tokens(
    gold_sent: conllu.TokenList,
    pred_sent: conllu.TokenList,
) -> tuple[list[tuple[conllu.Token, conllu.Token]], list[conllu.Token], list[conllu.Token]]:
    """Pair gold and predicted words that cover exactly the same characters.

    Arguments:
        gold_sent: A gold standard sentence.
        pred_sent: The corresponding predicted sentence.

    Returns:
        tuple(pairs, gold_only, pred_only) words with a one-to-one counterpart, then the words
        on each side left over by a split or merge, which have no comparable counterpart.

    Raises:
        ValueError: If the two sentences do not cover the same characters.

    """
    gold_spans = _character_spans(gold_sent)
    pred_spans = _character_spans(pred_sent)

    gold_stream = ''.join(''.join((t['form'] or '').split()) for t, _, _ in gold_spans)
    pred_stream = ''.join(''.join((t['form'] or '').split()) for t, _, _ in pred_spans)
    if gold_stream != pred_stream:
        sent_id = gold_sent.metadata.get('sent_id', '<no sent_id>')
        msg = f'Sentence {sent_id} does not cover the same characters in gold and prediction.'
        raise ValueError(msg)

    pred_by_span = {(start, end): token for token, start, end in pred_spans}
    matched_spans = set()
    pairs, gold_only = [], []
    for token, start, end in gold_spans:
        counterpart = pred_by_span.get((start, end))
        if counterpart is None:
            gold_only.append(token)
        else:
            pairs.append((token, counterpart))
            matched_spans.add((start, end))

    pred_only = [token for token, start, end in pred_spans if (start, end) not in matched_spans]
    return pairs, gold_only, pred_only


def align_sentences(
    gold_sentences: list[conllu.TokenList],
    pred_sentences: list[conllu.TokenList],
) -> list[tuple[conllu.TokenList, conllu.TokenList]]:
    """Pair gold and predicted sentences on `sent_id`, falling back to position.

    Arguments:
        gold_sentences: List of gold standard sentences.
        pred_sentences: List of predicted sentences.

    Returns:
        Pairs in gold order.

    Raises:
        ValueError: If sentences cannot be paired.

    """
    gold_ids = [sent.metadata.get('sent_id') for sent in gold_sentences]
    pred_ids = [sent.metadata.get('sent_id') for sent in pred_sentences]

    usable = all(gold_ids) and all(pred_ids) and len(set(pred_ids)) == len(pred_ids)
    if not usable:
        # no usable identifiers: position only, so counts must agree
        return list(zip(gold_sentences, pred_sentences, strict=True))

    by_id = dict(zip(pred_ids, pred_sentences, strict=True))
    missing = [sent_id for sent_id in gold_ids if sent_id not in by_id]
    if missing:
        msg = f'{len(missing)} gold sentence(s) absent from the predictions, first: {missing[0]}'
        raise ValueError(msg)

    return [(sent, by_id[sent_id]) for sent, sent_id in zip(gold_sentences, gold_ids, strict=True)]


def alignment_report(
    gold_sentences: list[conllu.TokenList],
    pred_sentences: list[conllu.TokenList],
) -> dict[str, Any]:
    """Summarise where a prediction's tokenisation disagrees with the gold standard.

    Arguments:
        gold_sentences: List of gold standard sentences.
        pred_sentences: List of predicted sentences.

    Returns:
        Counts of aligned and unaligned words alongside the affected sentence IDs.

    """
    aligned = gold_unaligned = pred_unaligned = 0
    affected = []
    for gold_sent, pred_sent in align_sentences(gold_sentences, pred_sentences):
        pairs, gold_only, pred_only = align_tokens(gold_sent, pred_sent)
        aligned += len(pairs)
        gold_unaligned += len(gold_only)
        pred_unaligned += len(pred_only)
        if gold_only or pred_only:
            affected.append(
                {
                    'sent_id': gold_sent.metadata.get('sent_id'),
                    'gold_only': [t['form'] for t in gold_only],
                    'pred_only': [t['form'] for t in pred_only],
                },
            )

    return {
        'sentences': len(gold_sentences),
        'aligned_tokens': aligned,
        'gold_only_tokens': gold_unaligned,
        'pred_only_tokens': pred_unaligned,
        'affected_sentences': affected,
    }


def compare_predictions(
    gold_sentences: list[conllu.TokenList],
    pred_sentences: list[conllu.TokenList],
    as_df: bool = True,  # noqa: FBT001
) -> pd.DataFrame | list[dict[str, Any]]:
    """Compare gold and predicted annotations.

    Arguments:
        gold_sentences: List of gold standard sentences.
        pred_sentences: List of predicted sentences.
        as_df: Whether to return the results as a DataFrame (default) or a list of dictionaries.

    Returns:
        Returns a DataFrame or a list of dicts with the following:
        - sent_id, token_id, form
        - gold values for: upos, xpos, feats, lemma, head, deprel
        - pred values for: upos, xpos, feats, lemma, head, deprel
        - correctness flags for: upos, xpos, feats, lemma, head, deprel

    """
    comparisons = []

    for idx, (gold_sent, pred_sent) in enumerate(align_sentences(gold_sentences, pred_sentences)):
        sent_id = gold_sent.metadata.get('sent_id', f'sent_{idx}')

        # words left unaligned by a tokenisation disagreement are dropped
        pairs, _, _ = align_tokens(gold_sent, pred_sent)
        for gold_token, pred_token in pairs:
            # ignore subtypes in deprel comparison
            gold_deprel = _normalize_deprel(gold_token['deprel']) if gold_token.get('deprel') else '_'
            pred_deprel = _normalize_deprel(pred_token['deprel']) if pred_token.get('deprel') else '_'

            comp = {
                'sent_id': sent_id,
                'token_id': gold_token['id'],
                'form': gold_token['form'],
                # gold values
                'gold_upos': gold_token['upos'],
                'gold_xpos': gold_token['xpos'],
                'gold_feats': feature_dict_to_string(gold_token['feats']) if gold_token['feats'] else '_',
                'gold_lemma': gold_token['lemma'],
                'gold_head': gold_token['head'],
                'gold_deprel': gold_deprel,
                # predicted values
                'pred_upos': pred_token['upos'],
                'pred_xpos': pred_token['xpos'],
                'pred_feats': feature_dict_to_string(pred_token['feats']) if pred_token['feats'] else '_',
                'pred_lemma': pred_token['lemma'],
                'pred_head': pred_token['head'],
                'pred_deprel': pred_deprel,
                # correctness flags
                'upos_correct': gold_token['upos'] == pred_token['upos'],
                'xpos_correct': gold_token['xpos'] == pred_token['xpos'],
                'feats_correct': gold_token['feats'] == pred_token['feats'],
                'lemma_correct': gold_token['lemma'] == pred_token['lemma'],
                'head_correct': gold_token['head'] == pred_token['head'],
                'deprel_correct': gold_deprel == pred_deprel,
            }
            comparisons.append(comp)

    return pd.DataFrame(comparisons) if as_df else comparisons


def _normalize_deprel(deprel: str) -> str:
    """Return deprel without any subtypes."""
    return deprel.split(':', maxsplit=1)[0] if deprel else deprel


def head_projection(pairs: list[tuple[conllu.Token, conllu.Token]]) -> dict[int, int]:
    """Map predicted token IDs to gold token IDs for a pair of aligned sentences."""
    return {pred_token['id']: gold_token['id'] for gold_token, pred_token in pairs}


def _projected_head(pred_token: conllu.Token, head_map: dict[int, int] | None) -> Any:
    """Return a predicted head expressed in gold token IDs."""
    head = pred_token['head']
    if head_map is None or head is None or head == 0:
        return head
    return head_map.get(head)


def _test_token_metric(
    gold_token: conllu.Token,
    pred_token: conllu.Token,
    metric: str,
    head_map: dict[int, int] | None = None,
) -> bool:
    """Test if a given token matches the gold standard for a specific metric."""
    if metric.lower() in ['upos', 'xpos', 'lemmas']:
        metric = 'lemma' if metric.lower() == 'lemmas' else metric.lower()
        return bool(gold_token[metric] == pred_token[metric])

    if metric.lower() == 'ufeats':
        gt_feats = gold_token.get('feats', {})
        pd_feats = pred_token.get('feats', {})
        if gt_feats and pd_feats:
            return gt_feats.keys() == pd_feats.keys() and all(gt_feats[k] == pd_feats[k] for k in gt_feats)
        return False

    if metric.lower() == 'alltags':
        return (
            _test_token_metric(gold_token, pred_token, 'upos')
            and _test_token_metric(gold_token, pred_token, 'xpos')
            and _test_token_metric(gold_token, pred_token, 'ufeats')
        )
    if metric.lower() == 'uas':
        return bool(gold_token['head'] == _projected_head(pred_token, head_map))

    if metric.lower() == 'las':
        return (gold_token['head'] == _projected_head(pred_token, head_map)) and (
            _normalize_deprel(gold_token['deprel']) == _normalize_deprel(pred_token['deprel'])
        )

    msg = f'Unknown metric: {metric}'
    raise ValueError(msg)


def sentence_score(
    gold_sentences: list[conllu.TokenList],
    pred_sentences: list[conllu.TokenList],
    metric: str,
) -> list[float]:
    """Compute per-sentence accuracy for a given metric.

    Arguments:
        gold_sentences: List of gold standard sentences.
        pred_sentences: List of predicted sentences.
        metric: Metric to compute (upos, xpos, lemmas, ufeats, alltags, uas, las)

    Returns:
        List of sentence-level accuracy scores.

    """
    sentence_scores = []

    for gold_sent, pred_sent in align_sentences(gold_sentences, pred_sentences):
        pairs, gold_only, _ = align_tokens(gold_sent, pred_sent)
        head_map = head_projection(pairs)

        # gold words the tokeniser split or merged away count against the score
        correct = sum(1 for gold_tok, pred_tok in pairs if _test_token_metric(gold_tok, pred_tok, metric, head_map))
        total = len(pairs) + len(gold_only)

        if total > 0:
            sentence_scores.append(correct / total)

    return sentence_scores


def bootstrap_test(  # noqa: PLR0913, PLR0917
    scores1: list[float],
    scores2: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    significance_level: float = 0.05,
    seed: int | None = 0,
) -> tuple[float, float, float, float, bool]:
    """Use bootstrap test to determine if difference between two sets of scores is significant.

    Arguments:
        scores1: List of scores from model 1.
        scores2: List of scores from model 2.
        n_bootstrap: Number of bootstrap samples to draw.
        confidence: Confidence level for the interval.
        significance_level: Significance level for the test.
        seed: Fixed by default so a reported confidence interval is reproducible.

    Returns:
        tuple(mean_diff, ci_lower, ci_upper, p_value, is_significant)

    """
    observed_diff = np.mean(scores1) - np.mean(scores2)
    rng = np.random.default_rng(seed)

    # bootstrap resampling
    bootstrap_diffs = []
    n = len(scores1)

    for _ in range(n_bootstrap):
        # resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        boot_scores1 = [scores1[i] for i in indices]
        boot_scores2 = [scores2[i] for i in indices]
        boot_diff = np.mean(boot_scores1) - np.mean(boot_scores2)
        bootstrap_diffs.append(boot_diff)

    # compute confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    # two-tailed p-value
    # proportion of bootstrap samples where diff has opposite sign
    if observed_diff > 0:
        p_value = np.sum(np.array(bootstrap_diffs) <= 0) / n_bootstrap
    else:
        p_value = np.sum(np.array(bootstrap_diffs) >= 0) / n_bootstrap

    p_value = 2 * min(p_value, 1 - p_value)  # two-tailed
    is_significant = p_value < significance_level

    return float(observed_diff), float(ci_lower), float(ci_upper), float(p_value), bool(is_significant)


def paired_t_test(
    scores1: list[float],
    scores2: list[float],
    significance_level: float = 0.05,
) -> tuple[float, float, float, bool]:
    """Use paired t-test to determine if difference between two sets of scores is significant.

    Arguments:
        scores1: List of scores from model 1.
        scores2: List of scores from model 2.
        significance_level: Significance level for the test

    """
    if len(scores1) != len(scores2):
        msg = 'Score lists must have the same length'
        raise ValueError(msg)

    # compute the t-statistic and p-value
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    mean_diff = np.mean(scores1) - np.mean(scores2)
    is_significant = p_value < significance_level

    return float(mean_diff), float(t_stat), float(p_value), bool(is_significant)


def track_error_resolution(key_1: str, key_2: str, metric: str, token_comparisons: pd.DataFrame) -> dict[str, Any]:
    """Track errors across two iterations.

    Arguments:
        key_1: Key for the first iteration.
        key_2: Key for the second iteration.
        metric: Metric to track (e.g., 'upos', 'xpos', 'lemmas', 'ufeats', 'alltags', 'uas', 'las')
        token_comparisons: Dictionary mapping iteration keys to token comparison DataFrames

    Returns:
        Dictionary with counts of errors fixed and present.

    """
    df1 = token_comparisons[key_1]
    df2 = token_comparisons[key_2]
    error_col = f'{metric}_correct'
    # merge on token_key
    merged = df1.merge(df2, on='token_key', suffixes=('_iter1', '_iter2'))
    # find tokens with errors in iter1
    iter1_errors = merged[~merged[f'{error_col}_iter1']]
    # amongst those with errors in iter1, count how many were fixed
    fixed = iter1_errors[iter1_errors[f'{error_col}_iter2']].shape[0]
    still_error = iter1_errors[~iter1_errors[f'{error_col}_iter2']].shape[0]

    return {
        'errors_in_iter1': len(iter1_errors),
        'fixed_in_iter2': fixed,
        'still_error_in_iter2': still_error,
        'fix_rate': fixed / len(iter1_errors) if len(iter1_errors) > 0 else 0,
    }


def classify_construction(sentence: conllu.TokenList) -> str:
    """Classify a sentence into one of several construction types.

    - Simple single object: "Item + NUM + NOUN" (basic enumeration)
    - Simple multi-object: "Item + NUM + NOUN + et + NUM/NOUN" (coordination without complexity)
    - Complex single object: "Item + NUM + NOUN + subclauses/modifiers" (relative clauses, acl, etc.)
    - Complex multi-object: "Item + NUM + NOUN + DET (quarum) + ..." (relative pronoun + continuation)
    - Code-switching: Contains Occitan tokens (detected via language classifier)

    Arguments:
        sentence: A conllu TokenList (parsed sentence).

    Returns:
        Construction type string.

    """
    # extract key features
    tokens = [tok for tok in sentence if isinstance(tok['id'], int)]
    if not tokens:
        return 'unknown'

    # check for code-switching
    has_occitan = any(get_token_language(tok)[0] == 'occitan' for tok in tokens)

    # check for "item" at start
    starts_with_item = tokens[0]['lemma'] and tokens[0]['lemma'].lower() == 'item'

    # extract structural features
    upos_seq = [tok['upos'] for tok in tokens]
    deprels = [tok['deprel'] for tok in tokens if tok['deprel']]

    # count key elements
    has_coordination = 'CCONJ' in upos_seq  # et, aut, etc.
    has_relative = any(deprel and deprel.startswith('acl') for deprel in deprels)  # relative clauses
    has_relative_pron = any(
        tok['lemma'] and tok['lemma'].lower() in ['qui', 'quae', 'quod', 'quarum'] for tok in tokens
    )

    # count number of nouns (objects)
    noun_count = sum(1 for upos in upos_seq if upos == 'NOUN')

    # classification logic
    if has_occitan:
        construction_type = 'code_switching'
    elif has_relative_pron and has_coordination and noun_count > 1:
        construction_type = 'complex_multi_object'
    elif has_relative or has_relative_pron:
        construction_type = 'complex_single_object'
    elif has_coordination and noun_count > 1:
        construction_type = 'simple_multi_object'
    elif starts_with_item and noun_count >= 1:
        construction_type = 'simple_single_object'
    else:
        construction_type = 'other'

    return construction_type


def compute_construction_accuracy(
    gold_sentences: list[conllu.TokenList],
    pred_sentences: list[conllu.TokenList],
    metric: str,
) -> dict[str, dict[str, Any]]:
    """Compute accuracy for each construction type.

    Arguments:
        gold_sentences: List of gold standard sentences.
        pred_sentences: List of predicted sentences.
        metric: Metric to compute (upos, xpos, lemmas, ufeats, alltags, uas, las).

    Returns:
        Dictionary mapping construction types to accuracy metrics.

    """
    # Initialize counters for each construction type
    construction_stats: dict[str, dict[str, Any]] = {}

    for gold_sent, pred_sent in align_sentences(gold_sentences, pred_sentences):
        # classify the sentence based on gold standard
        construction_type = classify_construction(gold_sent)

        # Initialize stats if first time seeing this type
        if construction_type not in construction_stats:
            construction_stats[construction_type] = {'correct': 0, 'total': 0, 'sentences': 0}

        construction_stats[construction_type]['sentences'] += 1

        # count correct tokens for this sentence
        # unaligned gold words count as errors
        pairs, gold_only, _ = align_tokens(gold_sent, pred_sent)
        head_map = head_projection(pairs)
        construction_stats[construction_type]['total'] += len(pairs) + len(gold_only)
        for gold_tok, pred_tok in pairs:
            if _test_token_metric(gold_tok, pred_tok, metric, head_map):
                construction_stats[construction_type]['correct'] += 1

    # compute accuracy for each construction type
    results = {}
    for construction_type, construction_data in construction_stats.items():
        accuracy = construction_data['correct'] / construction_data['total'] if construction_data['total'] > 0 else 0.0
        results[construction_type] = {
            'accuracy': accuracy,
            'correct': construction_data['correct'],
            'total': construction_data['total'],
            'sentences': construction_data['sentences'],
        }

    return results


def compute_training_stats(sentences: list[conllu.TokenList], name: str) -> dict[str, Any]:
    """Compute comprehensive statistics for a training set.

    Arguments:
        sentences: List of sentences in the training set.
        name: Name of the training set.

    Returns:
            Dictionary with various statistics:
            - name: Name of the training set.
            - sentences: Number of sentences.
            - tokens: Number of tokens.
            - unique_lemmas: Number of unique lemmas.
            - unique_forms: Number of unique word forms.
            - upos_counts: Dictionary with counts of each UPOS tag.
            - deprel_counts: Dictionary with counts of each dependency relation.
            - lemma_list: Set of unique lemmata.

    """
    # basic counts
    n_sentences = len(sentences)
    n_tokens = sum(len([t for t in sent if isinstance(t['id'], int)]) for sent in sentences)

    # vocabulary
    unique_lemmata = extract_unique_lemmata(sentences, include_propn=True)
    unique_forms = set()
    for sent in sentences:
        for token in sent:
            if isinstance(token['id'], int) and token['form']:
                unique_forms.add(token['form'].lower())

    # pos distribution
    upos_counts = extract_upos_counts(sentences)

    # deprels
    deprel_counts: dict[str, int] = {}
    for sent in sentences:
        for token in sent:
            if isinstance(token['id'], int) and token['deprel']:
                deprel = token['deprel'].split(':')[0]  # remove subtypes
                deprel_counts[deprel] = deprel_counts.get(deprel, 0) + 1

    return {
        'name': name,
        'sentences': n_sentences,
        'tokens': n_tokens,
        'unique_lemmas': len(unique_lemmata),
        'unique_forms': len(unique_forms),
        'upos_counts': upos_counts,
        'deprel_counts': deprel_counts,
        'lemma_list': unique_lemmata,
    }


def _get_sentence_context(
    sent_id: str,
    predicted: list[conllu.TokenList],
    gold: list[conllu.TokenList],
) -> tuple[conllu.TokenList | None, conllu.TokenList | None]:
    """Return full sentence context for an error."""
    for gold_sent, pred_sent in align_sentences(gold, predicted):
        if gold_sent.metadata.get('sent_id') == sent_id:
            return pred_sent, gold_sent
    return None, None


def get_error_example(
    error_row: pd.Series,
    pred_sentences: list[conllu.TokenList],
    gold_sentences: list[conllu.TokenList],
    description: str,
) -> list[str]:
    """Get a detailed error example."""
    error: list[str] = [
        f'\n{"=" * 70}',
        f'Example: {description}',
        f'{"=" * 70}',
    ]

    pred_sent, gold_sent = _get_sentence_context(error_row['sent_id'], pred_sentences, gold_sentences)

    if pred_sent and gold_sent:
        # show sentence text
        sent_text = ' '.join([t['form'] for t in pred_sent])
        error.append(f'\nSentence: {sent_text}')
        error.append(f'Sent ID: {error_row["sent_id"]}')

        # show the error token in context
        error.append(f"\nError token: '{error_row['form']}' (ID: {error_row['token_id']})")
        error.append(f'Error types: {error_row["error_types"]}')

        # show attribute comparisons
        if 'UPOS' in error_row['error_types']:
            error.append(f'  UPOS: {error_row["gold_upos"]} (gold) → {error_row["pred_upos"]} (predicted)')
        if 'Lemma' in error_row['error_types']:
            error.append(f'  Lemma: {error_row["gold_lemma"]} (gold) → {error_row["pred_lemma"]} (predicted)')
        if 'HEAD' in error_row['error_types']:
            # get head tokens
            gold_head_token = [t for t in gold_sent if t['id'] == error_row['gold_head']]
            pred_head_token = [t for t in pred_sent if t['id'] == error_row['pred_head']]
            gold_head_form = gold_head_token[0]['form'] if gold_head_token else 'ROOT'
            pred_head_form = pred_head_token[0]['form'] if pred_head_token else 'ROOT'
            error.append(
                f"  HEAD: {error_row['gold_head']} '{gold_head_form}' (gold)"
                f" → {error_row['pred_head']} '{pred_head_form}' (predicted)",
            )
        if 'DEPREL' in error_row['error_types']:
            error.append(f'  DEPREL: {error_row["gold_deprel"]} (gold) → {error_row["pred_deprel"]} (predicted)')

    return error
