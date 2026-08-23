"""Run the trained models over the gold standard and score them."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import (
    ANNOTATION_TIMES_PATH,
    ENSEMBLE_LEMMA_SUFFIX,
    GOLD_STANDARD_PATH,
    LATIN_MODEL_DIR,
    PROJECT_ROOT,
)
from .io import load_file

if TYPE_CHECKING:
    import conllu

EVALUATION_DIR = PROJECT_ROOT / 'data' / 'evaluation'

# base Stanza package for each configuration, and whether it injects the domain lexicon
CONFIGURATIONS: dict[str, tuple[str, bool]] = {
    'ittb': ('ittb', False),
    'ittb_ens': ('ittb', True),
    'llct': ('llct_ftv', False),
    'llct_ens': ('llct_ftv', True),
    'proiel': ('proiel', False),
    'proiel_ens': ('proiel', True),
    'perseus': ('perseus', False),
    'perseus_ens': ('perseus', True),
    'udante': ('udante', False),
    'udante_ens': ('udante', True),
    **{f'marseille_s{i}': (f'marseille_s{i}', True) for i in range(1, 11)},
}

# UDante is the *one* package that has a multi-word-token processor - oh, jolly good!
# so we run it through the full tokenizer instead of feeding it pre-tokenized text
# no hard feelings!
MWT_PACKAGES = frozenset({'udante'})


def patch_lemma_model(package: str, lexicon: list[list[str]], suffix: str) -> Path:
    """Write a copy of a base lemma model with 'lexicon' merged into its dictionaries.

    This is the "ensemble lemmatization" of the paper: Stanza consults 'word_dict' and
    'composite_dict' before falling back to the seq2seq model, so injecting domain entries makes
    them win.

    Arguments:
        package: Stanza package name, e.g. 'marseille_s9'.
        lexicon: '[form, lemma, upos]' triples.
        suffix: appended to the package name, e.g. '_custom'.

    Returns:
        The path written.

    """
    import torch  # noqa: PLC0415

    source = LATIN_MODEL_DIR / 'lemma' / f'{package}.pt'
    # weights_only=False is explicit because a Stanza lemma checkpoint carries its two Python
    # dicts alongside the tensors. Torch, for *some reason* is flipping this default,
    # and the flip would otherwise turn this into a hard failure
    model = torch.load(source, map_location='cpu', weights_only=False)
    word_dict, composite_dict = model['dicts']

    composite_dict.update({(form, upos): lemma for form, lemma, upos in lexicon})
    word_dict.update({form: lemma for form, lemma, _ in lexicon})

    destination = LATIN_MODEL_DIR / 'lemma' / f'{package}{suffix}.pt'
    torch.save(model, destination)
    return destination


def annotation_times() -> dict[str, float]:
    """Map model name to minutes spent annotating the seed batch that produced it."""
    with Path(ANNOTATION_TIMES_PATH).open(encoding='utf-8') as handle:
        return {row['resulting_model_name']: float(row['annotation_time_minutes']) for row in csv.DictReader(handle)}


def gold_input() -> tuple[str, list[dict[str, str]]]:
    """Return the gold standard as pipeline input text plus per-sentence metadata."""
    sentences: list[conllu.TokenList] = load_file(GOLD_STANDARD_PATH)
    texts, metadata = [], []
    for sentence in sentences:
        text = ' '.join(token['form'] for token in sentence)
        texts.append(text)
        metadata.append({'sent_id': sentence.metadata['sent_id'], 'text': text.replace(' .', '.')})
    return '\n\n'.join(texts), metadata


def run_configuration(
    name: str,
    package: str,
    *,
    lemma_suffix: str | None,
    feature_set: Any,
    output_dir: Path | None = None,
) -> Path:
    """Parse the gold standard with one configuration and write the output CoNLL-U.

    Arguments:
        name: configuration key, used in the output filename.
        package: Stanza package to load.
        lemma_suffix: if given, use the patched lemma model with this suffix.
        feature_set: language feature data from 'conllu_tools.io.load_language_data'.
        output_dir: where to write; defaults to 'data/evaluation/'.

    """
    import stanza  # noqa: PLC0415
    from conllu_tools.utils import feature_dict_to_string, normalize_morphology  # noqa: PLC0415

    from .io import write_stanza_document  # noqa: PLC0415

    text, metadata = gold_input()
    uses_mwt = package in MWT_PACKAGES

    options: dict[str, Any] = {
        'processors': 'tokenize,mwt,pos,lemma,depparse' if uses_mwt else 'tokenize,pos,lemma,depparse',
        'package': package,
        'download_method': None,
        'tokenize_no_ssplit': True,
    }
    if not uses_mwt:
        options['tokenize_pretokenized'] = True
    if lemma_suffix:
        options['lemma_model_path'] = str(LATIN_MODEL_DIR / 'lemma' / f'{package}{lemma_suffix}.pt')
        options['lemma_ensemble_dict'] = True

    doc = stanza.Pipeline('la', **options)(text)

    for sentence in doc.sentences:
        for word in sentence.words:
            xpos, feats = normalize_morphology(
                upos=word.upos,
                xpos=word.xpos,
                feats=word.feats,
                feature_set=feature_set,
            )
            word.xpos = xpos
            word.feats = feature_dict_to_string(feats)
            if word.head != '_' and word.deprel != '_':
                word.deps = f'{word.head}:{word.deprel}'

    destination = output_dir or EVALUATION_DIR
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f'la_{name}-ud-gold_test.conllu'
    write_stanza_document(doc, metadata, str(output))
    return output


def score(prediction_path: Path | str) -> dict[str, dict[str, float]]:
    """Score a prediction file against the gold standard with the CoNLL 2018 metrics."""
    from conllu_tools.evaluation import ConlluEvaluator  # noqa: PLC0415

    evaluator = ConlluEvaluator(eval_deprels=True, treebank_type='0')
    result = evaluator.evaluate_files(GOLD_STANDARD_PATH, str(prediction_path))
    return {
        key: {'precision': value.precision, 'recall': value.recall, 'f1': value.f1} for key, value in result.items()
    }


def evaluate_all(
    lemma_suffix: str = ENSEMBLE_LEMMA_SUFFIX,
    configurations: dict[str, tuple[str, bool]] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run and score every configuration.

    Arguments:
        lemma_suffix: which patched lemma models the ensemble configurations should use.
        configurations: defaults to 'CONFIGURATIONS'.
        output_dir: where to write prediction files; defaults to 'data/evaluation/'.

    """
    from conllu_tools.io import load_language_data  # noqa: PLC0415

    configurations = configurations or CONFIGURATIONS
    feature_set = load_language_data('feats', language='la', load_dalme=True)
    times = annotation_times()

    results: dict[str, Any] = {
        'generated': datetime.now(tz=UTC).isoformat(),
        'lemma_suffix': lemma_suffix,
        'data': {},
    }

    for name, (package, uses_lexicon) in configurations.items():
        print(f'  {name}...', flush=True)
        output = run_configuration(
            name,
            package,
            lemma_suffix=lemma_suffix if uses_lexicon else None,
            feature_set=feature_set,
            output_dir=output_dir,
        )
        results['data'][name] = {
            'output_path': str(output.relative_to(PROJECT_ROOT) if output.is_relative_to(PROJECT_ROOT) else output),
            'uses_lexicon': uses_lexicon,
            'evaluation': score(output),
        }
        if name in times:
            results['data'][name]['annotation_time_minutes'] = times[name]

    return results
