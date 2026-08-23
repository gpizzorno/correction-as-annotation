"""Build the ensemble lemma models, one per package, with the domain lexicon injected.

python scripts/build_ensemble_models.py --dry-run     # list what would be written
python scripts/build_ensemble_models.py               # write the master set
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bootstrapping.config import (
    ENSEMBLE_LEMMA_SUFFIX,
    LATIN_MODEL_DIR,
)
from bootstrapping.evaluation import CONFIGURATIONS, patch_lemma_model
from bootstrapping.lexicon import LEXICON_PATH, load_lexicon


def packages_needing_lexicon() -> list[str]:
    """Return the base Stanza packages that any ensemble configuration loads."""
    return sorted({package for package, uses_lexicon in CONFIGURATIONS.values() if uses_lexicon})


def main() -> None:
    """Patch every ensemble package's lemma model with the domain lexicon."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='report what would be written without writing anything')
    parser.add_argument('--packages', nargs='+', help='override the package list')
    args = parser.parse_args()

    suffix = ENSEMBLE_LEMMA_SUFFIX
    packages = args.packages or packages_needing_lexicon()

    lexicon = load_lexicon(LEXICON_PATH)
    print(f'lexicon : {LEXICON_PATH} ({len(lexicon):,} entries)')
    print(f'suffix  : {suffix}')
    print(f'models  : {LATIN_MODEL_DIR / "lemma"}\n')

    missing = [p for p in packages if not (Path(LATIN_MODEL_DIR) / 'lemma' / f'{p}.pt').exists()]

    if missing:
        third_party = [p for p in missing if not p.startswith('marseille_')]
        msg = [f'base lemma model(s) not found in {LATIN_MODEL_DIR / "lemma"}: {", ".join(missing)}']
        if third_party:
            packages_arg = ', '.join(f"'{p}'" for p in third_party)
            msg.append(
                'The baseline packages are third-party Stanza models. Fetch them with:\n'
                f"    python -c \"import stanza; [stanza.download('la', package=p, "
                f"processors='tokenize,pos,lemma,depparse') for p in [{packages_arg}]]\"",
            )
        if len(third_party) < len(missing):
            msg.append('The marseille_s* models are project-trained.')
        raise SystemExit('\n'.join(msg))

    for package in packages:
        destination = Path(LATIN_MODEL_DIR) / 'lemma' / f'{package}{suffix}.pt'
        if args.dry_run:
            state = 'overwrite' if destination.exists() else 'create'
            print(f'  [{state:9}] {destination}')
            continue
        print(f'  {package} -> {patch_lemma_model(package, lexicon, suffix).name}')

    if args.dry_run:
        print(f'\nDry run: {len(packages)} model(s) would be written.')


if __name__ == '__main__':
    main()
