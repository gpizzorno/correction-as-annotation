"""Parser functions for processing conllu data."""

from __future__ import annotations

from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any

import conllu
import pandas as pd
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler

from .config import LATIN_LEMMATA_PATH, OCCITAN_LEMMATA_PATH, REPORTED_ITERATIONS
from .io import load_file

DEFAULT_INDEX_LENGTHS = [2, 3, 4, 5, 6, 7, 8, 9]


def build_multi_prefix_index(
    lemmata: list[str],
    prefix_lengths: list[int] = DEFAULT_INDEX_LENGTHS,
) -> dict[int, dict[str, list[str]]]:
    """Build multiple prefix indices for different prefix lengths."""
    indices: dict[int, dict[str, list[str]]] = {}
    for prefix_len in prefix_lengths:
        indices[prefix_len] = defaultdict(list)
        for lemma in lemmata:
            if len(lemma) >= prefix_len:
                prefix = lemma[:prefix_len]
                indices[prefix_len][prefix].append(lemma)
    return indices


@cache
def language_reference(language: str) -> tuple[list[str], dict[int, dict[str, list[str]]]]:
    """Return the lemma list and prefix index for one language, loaded on first use.

    Arguments:
        language: 'latin' or 'occitan'.

    Returns:
        The lemmata, and the prefix index over them.

    Raises:
        FileNotFoundError: if the reference has not been built.

    """
    path = LATIN_LEMMATA_PATH if language == 'latin' else OCCITAN_LEMMATA_PATH
    if not Path(path).exists():
        msg = (
            f'{path} is missing. The language references are built locally, not shipped:\n'
            '    python -m bootstrapping.corpora\n'
            '    python -m bootstrapping.lemmata'
        )
        raise FileNotFoundError(msg)
    lemmata = load_file(path).split()
    return lemmata, build_multi_prefix_index(lemmata)


def resolve_tie(lemma: str, matches: list[list[str | float]]) -> list[str | float]:
    """Resolve ties using Demerau-Levenshtein similarity."""
    # check if all lemmata are the same
    if matches[0][1] == matches[1][1] and matches[0][1] == lemma:
        return matches[0]  # Latin

    # replace existing score
    for match in matches:
        match[2] = DamerauLevenshtein.normalized_similarity(lemma, match[1])

    # check for new ties
    if matches[0][2] == matches[1][2]:
        # still a tie, return first match
        return matches[0]  # Latin

    # return best match
    return max(matches, key=lambda x: x[2])


def fuzzy_match_adaptive(query: str, language: str, threshold: float) -> list[str | float | None]:  # noqa: C901
    """Find best match using adaptive prefix filtering based on query length.

    Arguments:
        query: The query string to match.
        language: The language of the query ('latin' or 'occitan').
        threshold: The similarity threshold for considering a match.

    Returns:
        A list containing the language, best matching lemma, and similarity score.

    """
    all_lemmata, lang_index = language_reference(language)

    # try straight matching first
    if query in all_lemmata:
        return [language, query, 1.0]

    target_prefix_length: int = round((len(query) - 3) * 0.4)
    candidates: list[str] = []

    # if target_prefix_length is at least 2, apply pre-filter
    if target_prefix_length >= 2:  # noqa: PLR2004
        # get the appropriate index
        if target_prefix_length not in lang_index:  # if not a direct fit
            target_prefix_length = min(
                lang_index.keys(),
                key=lambda x: abs(x - target_prefix_length),
            )  # get the closest down

        prefix_index = lang_index[target_prefix_length]
        prefix = query[:target_prefix_length]
        candidates = prefix_index.get(prefix, [])

        if not candidates:
            # try shorter prefixes if no exact match
            for shorter_len in range(target_prefix_length - 1, 2, -1):
                if shorter_len in lang_index:
                    shorter_prefix = query[:shorter_len]
                    sub_candidates = [
                        lemma
                        for p, lemmata in lang_index[shorter_len].items()
                        if p == shorter_prefix
                        for lemma in lemmata
                    ]
                    if sub_candidates:
                        candidates.extend(sub_candidates)

    if not candidates:
        candidates = all_lemmata

    best_match = None
    best_score = 0.0

    for candidate in candidates:
        jw_score = JaroWinkler.normalized_similarity(query, candidate, score_cutoff=0.4)
        dl_score = DamerauLevenshtein.normalized_similarity(query, candidate, score_cutoff=0.4)
        score = jw_score * 0.8 + dl_score * 0.2  # weight Jaro-Winkler slightly more
        if score > best_score:
            best_score = score
            best_match = candidate

    return [language, best_match, best_score] if best_score >= threshold else [language, None, 0.0]


def get_token_language(token: conllu.Token, threshold: float = 0.90) -> tuple[str, float]:
    """Determine language of a token.

    Arguments:
        token: A conllu Token.
        threshold: Confidence threshold for classification.

    Returns:
            Tuple with classification label and confidence score:
                'propn' if the token is a proper noun,
                'latin' if the token's lemma is in the Latin lemmata list,
                'occitan' if in the Occitan lemmata list,
                'unknown' otherwise.

    """
    if not isinstance(token, conllu.Token):
        msg = 'Input token must be a conllu Token.'
        raise TypeError(msg)

    upos = token.get('upos')
    if upos == 'PROPN':
        return ('propn', 1.0)

    lemma = token.get('lemma')

    if lemma:
        # adapt threshold based on lemma length
        if len(lemma) >= 10:  # noqa: PLR2004
            new_threshold = threshold - 0.20
        elif len(lemma) >= 6:  # noqa: PLR2004
            new_threshold = threshold - 0.10
        elif len(lemma) >= 4:  # noqa: PLR2004
            new_threshold = threshold - 0.05
        else:
            new_threshold = threshold

        matches: list[list[str | float]] = [  # format: language, match, score
            fuzzy_match_adaptive(lemma, 'latin', new_threshold),  # type: ignore [list-item]
            fuzzy_match_adaptive(lemma, 'occitan', new_threshold),  # type: ignore [list-item]
        ]

        # filter out non-matches
        matches = [m for m in matches if m[1] is not None]

        if matches:
            # check if all scores are zero
            if all(m[2] == 0.0 for m in matches):
                return ('unknown', 1.0)

            # check if both matches have the same score
            if len(matches) == 2 and matches[0][2] == matches[1][2]:  # noqa: PLR2004
                best_match = resolve_tie(lemma, matches)
            else:
                best_match = max(matches, key=lambda x: x[2])

            return (best_match[0], best_match[2])  # type: ignore [return-value]

    return ('unknown', 1.0)


def get_punctuation_token(idx: int) -> conllu.Token:
    """Return a full stop conllu Token."""
    return conllu.Token(
        {
            'id': idx,
            'form': '.',
            'lemma': '.',
            'upostag': 'PUNCT',
            'xpostag': 'u--------',
            'feats': '_',
            'head': '_',
            'deprel': '_',
            'deps': '_',
            'misc': 'Gloss=full_stop',
        },
    )


def extract_plot_data(metric: str, construction_results: dict[str, Any]) -> pd.DataFrame:
    """Convert nested construction results to DataFrame for plotting.

    Arguments:
        metric: The evaluation metric to extract (e.g., 'UPOS', 'UAS').
        construction_results: Nested dictionary with construction accuracy results.

    Returns:
        A pandas DataFrame suitable for plotting construction accuracies.

    """
    plot_data = []

    for iter_key, construction_accs in construction_results[metric].items():
        # find iteration name
        iter_name = next((it['name'] for it in REPORTED_ITERATIONS if it['key'] == iter_key), iter_key)

        for construction_type, stats in construction_accs.items():
            plot_data.append(
                {
                    'Iteration': iter_name,
                    'Construction Type': construction_type,
                    'Accuracy': stats['accuracy'] * 100,  # convert to percentage
                    'Sentences': stats['sentences'],
                    'Tokens': stats['total'],
                },
            )

    return pd.DataFrame(plot_data)
