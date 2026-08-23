"""Utils to build lemma inventories extracted from the comparison treebanks and the reference dictionaries.

Rebuild with:

    python -m bootstrapping.corpora
    python -m bootstrapping.lemmata

"""

from __future__ import annotations

import argparse
import json
from functools import cache
from pathlib import Path

from .config import (
    CIRCSE_LEMMATA_PATH,
    CLTK_LEMMATA_PATH,
    DOM_LEMMATA_PATH,
    ITTB_LEMMATA_PATH,
    LATIN_LEMMATA_PATH,
    LEWIS_LEMMATA_PATH,
    LLCT_LEMMATA_PATH,
    MARSEILLE_LEMMATA_PATH,
    OCCITAN_LEMMATA_PATH,
    OTTB_LEMMATA_PATH,
    PERSEUS_LEMMATA_PATH,
    PROIEL_LEMMATA_PATH,
    PROJECT_ROOT,
    UDANTE_LEMMATA_PATH,
)
from .corpora import corpus_path, missing_corpora
from .extractors import clean_lemma, extract_unique_lemmata
from .io import load_file

# raw dictionary sources. Large, third-party, and not committed
CLTK_SOURCE_PATH = PROJECT_ROOT / 'data' / 'reference' / 'latin_lemmata_cltk.json'
LEWIS_SOURCE_PATH = PROJECT_ROOT / 'data' / 'reference' / 'lewis.yaml'

# occitan lemmata contributed by DALME itself
DALME_OCCITAN_PATH = PROJECT_ROOT / 'data' / 'reference' / 'dalme_occitan_lemmata.txt'

# corpus code -> the inventory built from it
INVENTORIES: dict[str, str] = {
    'la_ittb': ITTB_LEMMATA_PATH,
    'la_llct': LLCT_LEMMATA_PATH,
    'la_perseus': PERSEUS_LEMMATA_PATH,
    'la_proiel': PROIEL_LEMMATA_PATH,
    'la_udante': UDANTE_LEMMATA_PATH,
    'la_circse': CIRCSE_LEMMATA_PATH,
    'oc_ttb': OTTB_LEMMATA_PATH,
    'la_marseille': MARSEILLE_LEMMATA_PATH,
}

# extracts of the raw dictionaries, on the language-reference rule
DICTIONARIES: dict[str, tuple[Path, str]] = {
    'cltk': (CLTK_SOURCE_PATH, CLTK_LEMMATA_PATH),
    'lewis': (LEWIS_SOURCE_PATH, LEWIS_LEMMATA_PATH),
}

# treebanks contributing to each language reference
LATIN_UNION_TREEBANKS = ('la_ittb', 'la_perseus', 'la_llct', 'la_proiel')
OCCITAN_UNION_TREEBANKS = ('oc_ttb',)

# plain word lists contributing to the Occitan reference
OCCITAN_UNION_WORDLISTS = (DOM_LEMMATA_PATH, str(DALME_OCCITAN_PATH))

# only 'la_marseille' is included locally
FETCHED = tuple(code for code in INVENTORIES if code != 'la_marseille')


@cache
def _treebank_lemmata(code: str, include_propn: bool) -> frozenset[str]:  # noqa: FBT001
    """Parse one treebank and extract its lemmata."""
    return frozenset(extract_unique_lemmata(load_file(str(corpus_path(code))), include_propn=include_propn))


def treebank_lemmata(code: str, *, include_propn: bool = True) -> set[str]:
    """Extract the distinct cleaned lemmata of one assembled treebank.

    Arguments:
        code: filename prefix, e.g. 'la_ittb'.
        include_propn: keep proper nouns. True for inventories, False for language references.

    Returns:
        The set of cleaned lemmata.

    """
    return set(_treebank_lemmata(code, include_propn))


def dictionary_lemmata(path: Path | str, *, include_propn: bool = True) -> set[str]:
    """Extract the cleaned lemmata of a raw dictionary source.

    Arguments:
        path: the '.json' or '.yaml' source.
        include_propn: keep capitalised headwords.

    Returns:
        The set of cleaned lemmata.

    Raises:
        FileNotFoundError: if the source is not present.

    """
    source = Path(path)
    if not source.exists():
        msg = f'dictionary source not found: {source}. It is third-party and not committed.'
        raise FileNotFoundError(msg)

    if source.suffix == '.json':
        headwords = json.loads(source.read_text(encoding='utf-8')).values()
    else:
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415  -- only for Lewis, an optional input

        headwords = yaml.safe_load(source.read_text(encoding='utf-8'))

    lemmata = {clean_lemma(headword) for headword in headwords if include_propn or not headword[:1].isupper()}
    lemmata.discard(None)
    return lemmata  # type: ignore[return-value]


def dictionary_reference(source: Path, extract: str) -> set[str]:
    """Return one dictionary's contribution to the Latin reference.

    Arguments:
        source: the raw '.json' or '.yaml' dictionary.
        extract: the committed extract derived from it.

    Returns:
        The set of lemmata.

    """
    path = Path(extract)
    if path.exists():
        return set(path.read_text(encoding='utf-8').split())
    return dictionary_lemmata(source, include_propn=False)


def latin_union(dictionaries: dict[str, set[str]] | None = None) -> set[str]:
    """Build the Latin language reference: the dictionaries plus the four Latin treebanks.

    Arguments:
        dictionaries: precomputed dictionary sets, to avoid re-reading them during a full build.

    Returns:
        The set of lemmata.

    """
    if dictionaries is None:
        dictionaries = {name: dictionary_reference(src, dst) for name, (src, dst) in DICTIONARIES.items()}
    lemmata: set[str] = set().union(*dictionaries.values())
    for code in LATIN_UNION_TREEBANKS:
        lemmata |= treebank_lemmata(code, include_propn=False)
    return lemmata


def occitan_union() -> set[str]:
    """Build the Occitan language reference."""
    lemmata: set[str] = set()
    for code in OCCITAN_UNION_TREEBANKS:
        lemmata |= treebank_lemmata(code, include_propn=False)
    for path in OCCITAN_UNION_WORDLISTS:
        lemmata |= set(load_file(str(path)).split())
    return lemmata


def write_list(path: Path | str, lemmata: set[str]) -> Path:
    """Write one lemma list, sorted, one per line."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(sorted(lemmata)) + '\n', encoding='utf-8')
    return destination


def build(*, dictionaries: bool = True) -> dict[str, set[str]]:
    """Derive every lemma list this module owns.

    Arguments:
        dictionaries: also re-derive the CLTK and Lewis extracts, which needs the raw sources.

    Returns:
        Output path to the derived set of lemmata.

    """
    derived: dict[str, set[str]] = {
        path: treebank_lemmata(code) for code, path in INVENTORIES.items() if corpus_path(code).exists()
    }

    dictionary_sets = {name: dictionary_reference(src, dst) for name, (src, dst) in DICTIONARIES.items()}
    if dictionaries:
        for name, (source, destination) in DICTIONARIES.items():
            if source.exists():
                dictionary_sets[name] = dictionary_lemmata(source, include_propn=False)
                derived[destination] = dictionary_sets[name]

    derived[LATIN_LEMMATA_PATH] = latin_union(dictionary_sets)
    derived[OCCITAN_LEMMATA_PATH] = occitan_union()
    return derived


def check(derived: dict[str, set[str]] | None = None) -> list[tuple[str, int, int, bool]]:
    """Compare each derived list against the committed file of the same name.

    Arguments:
        derived: output of 'build'. Computed if not given.

    Returns:
        One '(name, derived, committed, equal)' row per list, committed being -1 when no file
        exists yet.

    """
    derived = derived if derived is not None else build()

    rows = []
    for path, lemmata in derived.items():
        committed = Path(path)
        if committed.exists():
            existing = set(committed.read_text(encoding='utf-8').split())
            rows.append((committed.name, len(lemmata), len(existing), lemmata == existing))
        else:
            rows.append((committed.name, len(lemmata), -1, False))
    return rows


def main() -> None:
    """Rebuild the lemma lists, or report how they differ from what is committed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='compare against the committed lists without writing')
    parser.add_argument(
        '--no-dictionaries',
        action='store_true',
        help='skip the CLTK and Lewis extracts and the Latin union, which need the raw sources',
    )
    args = parser.parse_args()

    missing = [code for code in missing_corpora() if code in FETCHED]
    if missing:
        print(f'missing treebanks: {", ".join(missing)}')
        print('run  python -m bootstrapping.corpora  first\n')

    derived = build(dictionaries=not args.no_dictionaries)

    if args.check:
        rows = check(derived)
        width = max(len(name) for name, *_ in rows)
        for name, got, committed, equal in rows:
            if committed < 0:
                print(f'  {name:<{width}}  {got:>6}  (not committed)')
                continue
            if equal:
                mark = 'ok'
            elif got == committed:
                mark = 'same count, different members'
            else:
                mark = f'differs by {abs(got - committed)}'
            print(f'  {name:<{width}}  {got:>6}  committed {committed:>6}  {mark}')
        differing = [name for name, _, _, equal in rows if not equal]
        print(f'\n{len(rows) - len(differing)}/{len(rows)} match')
        return

    for path, lemmata in derived.items():
        written = write_list(path, lemmata)
        print(f'  {written.name:<28} {len(lemmata):>6} lemmata')
    print(f'\n{len(derived)} list(s) written to {Path(LATIN_LEMMATA_PATH).parent}')


if __name__ == '__main__':
    main()
