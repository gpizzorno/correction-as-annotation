"""Shared data loading and frame building for the learning-dynamics analyses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bootstrapping.config import (
    EVALUATION_DATA_PATHS,
    EVALUATION_RESULTS_PATH,
    GOLD_STANDARD_PATH,
    REPORTED_SEED_KEYS,
)
from bootstrapping.io import load_file

if TYPE_CHECKING:
    import conllu
    import pandas as pd

# the baseline the bootstrapping chain starts from
BASELINE_KEY = 'ittb'


def iteration_keys(*, with_baseline: bool = False) -> list[str]:
    """Return the reported iteration keys, optionally with the ITTB baseline in front."""
    keys = list(REPORTED_SEED_KEYS)
    return [BASELINE_KEY, *keys] if with_baseline else keys


def iteration_number(key: str) -> int:
    """Return the iteration number from a key, e.g. 'marseille_s7' -> 7."""
    return int(key.split('_s')[1])


def load_gold() -> list[conllu.TokenList]:
    """Load the 200-sentence gold standard."""
    return load_file(GOLD_STANDARD_PATH)  # type: ignore[no-any-return]


def load_predictions(keys: list[str] | None = None) -> dict[str, list[conllu.TokenList]]:
    """Load each model's predictions over the gold standard, keyed by model."""
    return {key: load_file(EVALUATION_DATA_PATHS[key]) for key in (keys or iteration_keys())}


def token_comparisons(
    keys: list[str] | None = None,
    gold: list[conllu.TokenList] | None = None,
    predictions: dict[str, list[conllu.TokenList]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return a token-level gold/prediction comparison frame per model.

    Arguments:
        keys: which models to compare. Defaults to the reported iterations.
        gold: the gold standard, loaded if not supplied.
        predictions: predictions per model, loaded if not supplied.

    """
    from bootstrapping.analysis import compare_predictions  # noqa: PLC0415

    keys = keys or iteration_keys()
    gold = gold if gold is not None else load_gold()
    predictions = predictions if predictions is not None else load_predictions(keys)
    return {key: compare_predictions(gold, predictions[key]) for key in keys}


def evaluation_frame(keys: list[str] | None = None) -> pd.DataFrame:
    """Per-iteration metrics from 'results.json', one row per model."""
    import pandas as pd  # noqa: PLC0415

    results = load_file(EVALUATION_RESULTS_PATH)
    rows = []
    for key in keys or iteration_keys():
        entry = results['data'][key]
        row: dict[str, Any] = {
            'iteration': iteration_number(key) if '_s' in key else 0,
            'iteration_key': key,
        }
        if 'annotation_time_minutes' in entry:
            row['annotation_time_minutes'] = entry['annotation_time_minutes']
        for metric, values in entry['evaluation'].items():
            row[f'{metric}_f1'] = values['f1']
            row[f'{metric}_precision'] = values['precision']
            row[f'{metric}_recall'] = values['recall']
        rows.append(row)
    return pd.DataFrame(rows)


def token_accuracy(comparisons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-token accuracy per iteration, as percentages, from comparison frames."""
    import pandas as pd  # noqa: PLC0415

    rows = []
    for key, frame in comparisons.items():
        rows.append(
            {
                'iteration': iteration_number(key),
                'iteration_key': key,
                'total_tokens': len(frame),
                'upos_acc': frame['upos_correct'].mean() * 100,
                'xpos_acc': frame['xpos_correct'].mean() * 100,
                'feats_acc': frame['feats_correct'].mean() * 100,
                'lemma_acc': frame['lemma_correct'].mean() * 100,
                'head_acc': frame['head_correct'].mean() * 100,
                'deprel_acc': frame['deprel_correct'].mean() * 100,
            },
        )
    return pd.DataFrame(rows).sort_values('iteration')


def oov_rate(
    training_lemmata: set[str],
    gold_sentences: list[conllu.TokenList],
    *,
    include_propn: bool = False,
) -> dict[str, float]:
    """Share of gold lemma types and tokens absent from a training vocabulary."""
    from bootstrapping.extractors import extract_unique_lemmata  # noqa: PLC0415

    gold_types = extract_unique_lemmata(gold_sentences, include_propn=include_propn)
    unseen_types = {lemma for lemma in gold_types if lemma not in training_lemmata}

    total_tokens = unseen_tokens = 0
    for sentence in gold_sentences:
        for token in sentence:
            if not isinstance(token['id'], int) or not token['lemma']:
                continue
            if not include_propn and token['upos'] == 'PROPN':
                continue
            total_tokens += 1
            unseen_tokens += token['lemma'] not in training_lemmata

    return {
        'gold_types': len(gold_types),
        'oov_types': len(unseen_types),
        'oov_type_rate': len(unseen_types) / len(gold_types) if gold_types else 0.0,
        'gold_tokens': total_tokens,
        'oov_tokens': unseen_tokens,
        'oov_token_rate': unseen_tokens / total_tokens if total_tokens else 0.0,
    }


def annotation_hours(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add per-iteration and cumulative annotation hours to an evaluation frame."""
    frame = frame if frame is not None else evaluation_frame()
    frame = frame.copy()
    frame['annotation_time_hours'] = frame['annotation_time_minutes'] / 60
    frame['cumulative_time_hours'] = frame['annotation_time_hours'].cumsum()
    return frame
