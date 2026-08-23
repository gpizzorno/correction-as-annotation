"""Utility functions for extracting information from data."""

from __future__ import annotations

from collections import Counter

import conllu
import regex as re


def clean_lemma(lemma: str) -> str | None:
    """Clean a lemma."""
    if not lemma or not isinstance(lemma, str):
        return None
    # characters to remove
    lemma = lemma.lower()  # make lowercase
    lemma = re.sub(r'[\d]', '', lemma)  # remove digits
    lemma = re.sub(r'[-\.,\"\'\(\)\[\]\{\}\?:;_\+\*†°!\\«»øØ“”…–€]', '', lemma)  # remove other chars  # noqa: RUF001
    if len(lemma) == 0:
        return None
    lemma = lemma.strip()
    return lemma if len(lemma) else None


def extract_unique_lemmata(sentences: list[conllu.TokenList], include_propn: bool = True) -> set[str]:  # noqa: FBT001
    """Extract unique lemmas from sentences."""
    lemmata = set()
    for sentence in sentences:
        for token in sentence:
            if not isinstance(token, conllu.models.Token):
                continue

            if not include_propn and token['upos'] == 'PROPN':
                continue

            if isinstance(token.get('id'), int) and token.get('lemma'):
                clean = clean_lemma(token['lemma'])
                if clean:
                    lemmata.add(clean)
    return lemmata


def extract_upos_counts(sentences: list[conllu.TokenList] | conllu.TokenList) -> dict[str, int]:
    """Extract counts of each UPOS tag from sentences."""
    if isinstance(sentences, conllu.TokenList):
        sentences = [sentences]

    upos_counts: dict[str, int] = Counter()
    for sentence in sentences:
        for token in sentence:
            if not isinstance(token, conllu.models.Token):
                continue
            upos_counts[token['upos']] += 1
    return upos_counts
