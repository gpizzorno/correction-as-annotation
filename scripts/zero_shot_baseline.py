"""Zero-shot evaluation of current Stanza Latin models on the DALME-Marseille gold standard.

Must be run in a separate virtualenv.

    python -m venv .venv-modern
    .venv-modern/bin/pip install "stanza>=1.10" conllu conllu_tools
    .venv-modern/bin/python scripts/zero_shot_baseline.py --download

Models download to '$STANZA_RESOURCES_DIR' instead of the default '~/stanza_resources'), to set:

    export STANZA_RESOURCES_DIR=~/stanza_resources_modern
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import conllu
import stanza
from conllu_tools.evaluation import ConlluEvaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_FILE = PROJECT_ROOT / 'data' / 'gold' / 'la_marseille-ud-gold.conllu'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'evaluation' / 'modern'
RESULTS_FILE = OUTPUT_DIR / 'zero_shot_results.json'

# the study used these five treebank packages-kept so the rows line up
PACKAGES = ['ittb', 'llct', 'perseus', 'proiel', 'udante']

METRICS = ['UPOS', 'XPOS', 'UFeats', 'Lemmas', 'AllTags', 'UAS', 'LAS', 'CLAS', 'MLAS', 'BLEX', 'ELAS']


def load_gold(path: Path) -> tuple[str, list[dict[str, str]]]:
    """Return the gold text as pre-tokenized input and per-sentence metadata."""
    sentences = conllu.parse(path.read_text(encoding='utf-8'))
    texts, metadata = [], []
    for sentence in sentences:
        text = ' '.join(token['form'] for token in sentence if isinstance(token['id'], int))
        texts.append(text)
        metadata.append({'sent_id': sentence.metadata['sent_id'], 'text': text.replace(' .', '.')})
    return '\n\n'.join(texts), metadata


def write_conllu(doc: stanza.Document, metadata: list[dict[str, str]], path: Path) -> None:
    """Write a Stanza document to CoNLL-U, restoring sent_id/text metadata."""
    from stanza.utils.conll import CoNLL  # noqa: PLC0415

    sentences = CoNLL.convert_dict(doc.to_dict())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for sentence, meta in zip(sentences, metadata, strict=True):
            handle.write(f'# sent_id = {meta["sent_id"]}\n')
            handle.write(f'# text = {meta["text"]}\n')
            for line in sentence:
                handle.write('\t'.join(line) if isinstance(line, list) else str(line))
                handle.write('\n')
            handle.write('\n')


def run(packages: list[str], *, download: bool) -> dict[str, object]:
    """Evaluate each package on the gold standard and return results."""
    from conllu_tools.io import load_language_data  # noqa: PLC0415
    from conllu_tools.utils import feature_dict_to_string, normalize_morphology  # noqa: PLC0415

    feature_set = load_language_data('feats', language='la', load_dalme=True)
    text, metadata = load_gold(GOLD_FILE)
    evaluator = ConlluEvaluator(eval_deprels=True, treebank_type='0')
    results: dict[str, object] = {
        'generated': datetime.now(tz=UTC).isoformat(),
        'stanza_version': stanza.__version__,
        'data': {},
    }

    for package in packages:
        print(f'--- {package}')
        if download:
            stanza.download('la', package=package, processors='tokenize,pos,lemma,depparse')

        pipeline = stanza.Pipeline(
            'la',
            processors='tokenize,pos,lemma,depparse',
            package=package,
            tokenize_pretokenized=True,
            tokenize_no_ssplit=True,
            download_method=None,
        )
        doc = pipeline(text)

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

        output = OUTPUT_DIR / f'la_{package}_modern-ud-gold_test.conllu'
        write_conllu(doc, metadata, output)

        scored = evaluator.evaluate_files(str(GOLD_FILE), str(output))
        results['data'][package] = {  # type: ignore[index]
            'output_path': str(output.relative_to(PROJECT_ROOT)),
            'evaluation': {
                key: {'precision': value.precision, 'recall': value.recall, 'f1': value.f1}
                for key, value in scored.items()
            },
        }

    return results


def report(results: dict[str, object], baseline_path: Path) -> None:
    """Print current and previous results side by side."""
    old = json.loads(baseline_path.read_text())['data'] if baseline_path.exists() else {}
    header = f'{"package":10}{"":6}' + ''.join(f'{metric:>9}' for metric in METRICS)
    print(f'\n{header}')
    for package, entry in results['data'].items():  # type: ignore[attr-defined]
        new_scores = entry['evaluation']
        print(f'{package:10}' + ''.join(f'{new_scores[m]["f1"] * 100:9.1f}' for m in METRICS))
        if package in old:
            old_scores = old[package]['evaluation']
            print(f'{"":10}' + ''.join(f'{old_scores[m]["f1"] * 100:9.1f}' for m in METRICS))
            print(
                f'{"":10}{"Δ":6}'
                + ''.join(f'{(new_scores[m]["f1"] - old_scores[m]["f1"]) * 100:+9.1f}' for m in METRICS),
            )


def main() -> None:
    """Run the zero-shot evaluation and write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', nargs='+', default=PACKAGES, help='Stanza Latin packages to evaluate')
    parser.add_argument('--download', action='store_true', help='download each package before running')
    args = parser.parse_args()

    results = run(args.packages, download=args.download)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=4))
    print(f'\nWrote {RESULTS_FILE.relative_to(PROJECT_ROOT)}')
    report(results, PROJECT_ROOT / 'data' / 'evaluation' / 'results.json')


if __name__ == '__main__':
    main()
