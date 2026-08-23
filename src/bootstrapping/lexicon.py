"""Tools for building domain lemma lexicon injected into Stanza's lemmatizer."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from .config import GOLD_STANDARD_PATH, PROJECT_ROOT
from .io import load_file

if TYPE_CHECKING:
    import conllu


# the master: gold-free, and what everything reads by default
LEXICON_PATH = PROJECT_ROOT / 'data' / 'marseille_lemmata.json'

# the input the master is derived from by subtraction, not a superseded copy of it
SOURCE_LEXICON_PATH = PROJECT_ROOT / 'data' / 'marseille_lemmata_with_gold.json'
FMP_CORPUS_PATH = PROJECT_ROOT / 'data' / 'marseille_test_corpus.csv'

# pos values the manual-correction overlay left as raw DALME tags rather than UD ones
RAW_DALME_POS = frozenset(
    {'adjective', 'adposition', 'adverb', 'gerund', 'noun', 'numeral', 'proper noun', 'verb'},
)

# one [form, lemma, upos] triple
Entry = list[str]


def load_lexicon(path: Path | str = LEXICON_PATH) -> list[Entry]:
    """Load a lexicon file as a list of '[form, lemma, upos]' triples."""
    return json.loads(Path(path).read_text(encoding='utf-8'))  # type: ignore[no-any-return]


def as_lookup(entries: list[Entry]) -> dict[str, tuple[str, str]]:
    """Return '{form: (lemma, upos)}'. Forms are unique in the committed lexicon."""
    return {form: (lemma, upos) for form, lemma, upos in entries}


def manual_fix_forms(entries: list[Entry] | None = None) -> set[str]:
    """Return the forms that came from the manual-correction overlay."""
    entries = entries if entries is not None else load_lexicon()
    return {form for form, _, upos in entries if upos in RAW_DALME_POS}


def form_attestations(path: Path | str = FMP_CORPUS_PATH) -> dict[str, set[str]]:
    """Map each lowercased surface form to the set of corpus line IDs it occurs in."""
    attestations = defaultdict(set)
    with Path(path).open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            form = (row['word_form'] or '').strip().lower()
            if form:
                attestations[form].add(row['line_id'])
    return dict(attestations)


def gold_sentence_ids(path: str = GOLD_STANDARD_PATH) -> set[str]:
    """Return the sentence IDs of the evaluation gold standard."""
    sentences: list[conllu.TokenList] = load_file(path)
    return {s.metadata['sent_id'] for s in sentences}


def classify_entries(
    entries: list[Entry] | None = None,
    attestations: dict[str, set[str]] | None = None,
    gold_ids: set[str] | None = None,
) -> dict[str, list[str]]:
    """Split lexicon forms by whether they leak the gold standard.

    Arguments:
        entries: list of entries to classify.
        attestations: form attestations.
        gold_ids: list of gold sentence ids.

    Returns:
        A mapping with three keys:
            'elsewhere': Attested outside the gold set. Safe to keep.
            'gold_only': Attested only in gold sentences. These are the leakage.
            'unattested': Not in the database export at all (manual fixes).

    """
    entries = entries if entries is not None else load_lexicon(SOURCE_LEXICON_PATH)
    attestations = attestations if attestations is not None else form_attestations()
    gold_ids = gold_ids if gold_ids is not None else gold_sentence_ids()

    groups: dict[str, list[str]] = {'elsewhere': [], 'gold_only': [], 'unattested': []}
    for form, _, _ in entries:
        lines = attestations.get(form)
        if lines is None:
            groups['unattested'].append(form)
        elif lines <= gold_ids:
            groups['gold_only'].append(form)
        else:
            groups['elsewhere'].append(form)
    return groups


def build_clean_lexicon(entries: list[Entry] | None = None) -> tuple[list[Entry], list[str]]:
    """Return the lexicon with gold-only entries removed, plus the dropped forms."""
    entries = entries if entries is not None else load_lexicon(SOURCE_LEXICON_PATH)
    groups = classify_entries(entries)
    dropped = set(groups['gold_only'])
    kept = [entry for entry in entries if entry[0] not in dropped]
    return kept, sorted(dropped)


def write_clean_lexicon(path: Path | str = LEXICON_PATH) -> tuple[Path, list[str]]:
    """Rebuild the master lexicon from the source one and write it."""
    kept, dropped = build_clean_lexicon()
    destination = Path(path)
    destination.write_text(json.dumps(kept, indent=4) + '\n', encoding='utf-8')
    return destination, dropped


def gold_coverage(entries: list[Entry], gold_path: str = GOLD_STANDARD_PATH) -> dict[str, float]:
    """Measure how much of the gold standard a lexicon can answer by lookup alone."""
    lookup = as_lookup(entries)
    sentences: list[conllu.TokenList] = load_file(gold_path)

    total = in_lexicon = exact = 0
    for sentence in sentences:
        for token in sentence:
            if not isinstance(token['id'], int) or token['upos'] == 'PUNCT':
                continue
            total += 1
            hit = lookup.get(token['form'].lower())
            if hit is not None:
                in_lexicon += 1
                exact += hit[0] == token['lemma']

    return {
        'words': total,
        'coverage': in_lexicon / total * 100,
        'exact_lemma': exact / total * 100,
    }


def main() -> None:
    """Rebuild the master lexicon from the source one."""
    destination, dropped = write_clean_lexicon()
    print(f'Wrote {destination}')
    print(f'Dropped {len(dropped)} gold-only entries')


if __name__ == '__main__':
    main()
