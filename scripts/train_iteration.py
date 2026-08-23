"""Build the UD treebank for one bootstrapping iteration and emit its Stanza training commands.

Prints the commands by default and only runs them if 'execute` is passed.

    python scripts/train_iteration.py 3
    python scripts/train_iteration.py 3 --execute
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bootstrapping.config import PROJECT_ROOT
from bootstrapping.training import build_ud_dirs, default_udbase, training_commands


def main() -> int:
    """Build the treebank for one iteration and emit/run its training commands."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('iteration', type=int, help='iteration to train (1-9)')
    parser.add_argument('--udbase', type=Path, default=default_udbase(), help='where to write treebanks')
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'data' / 'stanza')
    parser.add_argument('--wordvec-dir', type=Path, default=PROJECT_ROOT / 'extern_data' / 'wordvec')
    parser.add_argument('--split-seed', type=int, default=0, help='RNG seed for the 90/10 shuffle')
    parser.add_argument('--skip-build', action='store_true', help='reuse existing treebank directories')
    parser.add_argument('--execute', action='store_true', help='run the commands')
    args = parser.parse_args()

    if not args.skip_build:
        written = build_ud_dirs(args.udbase, max_iteration=args.iteration, seed=args.split_seed)
        print(f'Built {written[args.iteration]}')

    commands = training_commands(
        args.iteration,
        udbase=args.udbase,
        data_root=args.data_root,
        wordvec_dir=args.wordvec_dir,
    )

    if not args.execute:
        print('\nTraining commands:\n')
        for command in commands:
            print(f'  {command}\n')
        print('Use --execute to run them.')
        return 0

    for command in commands:
        print(f'\n$ {command}')
        result = subprocess.run(command, shell=True, check=False)
        if result.returncode != 0:
            print(f'FAILED with exit code {result.returncode}', file=sys.stderr)
            return result.returncode

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
