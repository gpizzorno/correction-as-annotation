"""Parse the gold standard with every configuration and score the output.

python scripts/evaluate_models.py # master run -> data/evaluation/results.json

Running the master configuration overwrites both the prediction files and 'results.json',
so it is guarded by '--force' unless writing somewhere else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bootstrapping.config import (
    ENSEMBLE_LEMMA_SUFFIX,
    EVALUATION_RESULTS_PATH,
)
from bootstrapping.evaluation import CONFIGURATIONS, evaluate_all

METRICS = ['UPOS', 'XPOS', 'UFeats', 'Lemmas', 'AllTags', 'UAS', 'LAS', 'CLAS', 'MLAS', 'BLEX']

# smallest delta the two-decimal report can show
# anything under this rounds to +0.00.
REPORTABLE_DELTA = 0.005


def report(results: dict[str, Any], baseline: Path) -> None:
    """Print the scores and the delta against an existing results file."""
    old = json.loads(baseline.read_text())['data'] if baseline.exists() else {}
    print('\n' + f'{"configuration":18}' + ''.join(f'{m:>9}' for m in METRICS))
    for name, entry in results['data'].items():
        scores = entry['evaluation']
        print(f'{name:18}' + ''.join(f'{scores[m]["f1"] * 100:9.2f}' for m in METRICS))
        if name in old:
            deltas = [(scores[m]['f1'] - old[name]['evaluation'][m]['f1']) * 100 for m in METRICS]
            if any(abs(delta) >= REPORTABLE_DELTA for delta in deltas):
                print(f'{"  Δ vs existing":18}' + ''.join(f'{delta:+9.2f}' for delta in deltas))


def main() -> None:
    """Run every configuration and write the results file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--configurations', nargs='+', help='run only specific configuration keys')
    parser.add_argument('--output-dir', type=Path, help='write predictions to target instead of data/evaluation/')
    parser.add_argument('--results', type=Path, help='write the results JSON file here')
    parser.add_argument('--force', action='store_true', help='allow overwriting the master JSON file')
    args = parser.parse_args()

    suffix = ENSEMBLE_LEMMA_SUFFIX
    configurations = CONFIGURATIONS

    if args.configurations:
        unknown = [key for key in args.configurations if key not in CONFIGURATIONS]
        if unknown:
            msg = f'unknown configuration(s): {", ".join(unknown)}'
            raise SystemExit(msg)
        configurations = {key: CONFIGURATIONS[key] for key in args.configurations}

    results_path = args.results or (
        Path(EVALUATION_RESULTS_PATH) if args.output_dir is None else args.output_dir / 'results.json'
    )

    if results_path == Path(EVALUATION_RESULTS_PATH) and results_path.exists() and not args.force:
        msg = f'{results_path} exists. You can pass --force to overwrite, or --output-dir to write elsewhere'
        raise SystemExit(msg)

    print(f'lemma suffix   : {suffix}')
    print(f'configurations : {len(configurations)}')
    print(f'results        : {results_path}\n')

    results = evaluate_all(lemma_suffix=suffix, configurations=configurations, output_dir=args.output_dir)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=4))
    print(f'\nWrote {results_path}')
    report(results, Path(EVALUATION_RESULTS_PATH))


if __name__ == '__main__':
    main()
