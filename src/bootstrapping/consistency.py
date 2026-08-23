"""Intra-annotator consistency on repeated constructions."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, NamedTuple

from .config import GOLD_STANDARD_PATH, REPORTED_SEED_KEYS, SEED_TRAINING_PATHS
from .io import load_file

if TYPE_CHECKING:
    import conllu

MIN_SEQUENCE_LENGTH = 2

# a sequence must recur at least this many times
# across at least this many sessions, to count
MIN_OCCURRENCES = 2
MIN_SESSIONS = 2


class Repeat(NamedTuple):
    """A word sequence annotated more than once, in different sessions."""

    forms: tuple[str, ...]
    occurrences: list[tuple[str, Any]]  # (session label, sentence)


def annotated_sentences() -> list[tuple[str, conllu.TokenList]]:
    """Return every human-corrected sentence, tagged with the session that produced it."""
    items: list[tuple[str, conllu.TokenList]] = []
    for key in REPORTED_SEED_KEYS:
        items += [(key.removeprefix('marseille_'), s) for s in load_file(SEED_TRAINING_PATHS[key])]
    items += [('gold', s) for s in load_file(GOLD_STANDARD_PATH)]
    return items


def words(sentence: conllu.TokenList) -> list[Any]:
    """Return the non-punctuation word tokens of a sentence."""
    return [t for t in sentence if isinstance(t['id'], int) and t['upos'] != 'PUNCT']


def _key(sentence: conllu.TokenList) -> tuple[str, ...]:
    return tuple(t['form'].lower() for t in words(sentence))


def find_repeats(
    items: list[tuple[str, conllu.TokenList]] | None = None,
    *,
    cross_session_only: bool = True,
) -> list[Repeat]:
    """Find word sequences that were annotated more than once.

    Arguments:
        items: output of :func:'annotated_sentences'.
        cross_session_only: keep only sequences repeated across *different* batches, so the
            annotator cannot have been working from short-term memory of the earlier decision.

    """
    items = items if items is not None else annotated_sentences()

    grouped: dict[tuple[str, ...], list[tuple[str, Any]]] = defaultdict(list)
    for session, sentence in items:
        key = _key(sentence)
        if len(key) >= MIN_SEQUENCE_LENGTH:
            grouped[key].append((session, sentence))

    repeats = []
    for key, occurrences in grouped.items():
        if len(occurrences) < MIN_OCCURRENCES:
            continue
        if cross_session_only and len({s for s, _ in occurrences}) < MIN_SESSIONS:
            continue
        repeats.append(Repeat(key, occurrences))
    return repeats


def _annotation(sentence: conllu.TokenList) -> tuple[tuple[str, Any, str], ...]:
    return tuple((t['upos'], t['head'], t['deprel']) for t in words(sentence))


def score_repeats(repeats: list[Repeat]) -> dict[str, Any]:
    """Compare each repeat against its first occurrence, token by token.

    Arguments:
        repeats: list of repeated tokens.

    Returns:
        agreement rates with Wilson-free normal-approximation confidence intervals, plus
        Cohen's kappa for UPoS.

    """
    totals: Counter[str] = Counter()
    agreed: Counter[str] = Counter()
    upos_pairs: list[tuple[str, str]] = []
    divergent: list[Repeat] = []

    for repeat in repeats:
        reference = repeat.occurrences[0][1]
        if any(_annotation(other) != _annotation(reference) for _, other in repeat.occurrences[1:]):
            divergent.append(repeat)

        for _, other in repeat.occurrences[1:]:
            for a, b in zip(words(reference), words(other), strict=True):
                for field in ('upos', 'head', 'deprel'):
                    totals[field] += 1
                    agreed[field] += a[field] == b[field]
                totals['las'] += 1
                agreed['las'] += a['head'] == b['head'] and a['deprel'] == b['deprel']
                upos_pairs.append((a['upos'], b['upos']))

    def rate(field: str) -> dict[str, float]:
        n = totals[field]
        if not n:
            return {'agreement': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'), 'n': 0}
        p = agreed[field] / n
        margin = 1.96 * math.sqrt(p * (1 - p) / n)
        return {
            'agreement': p * 100,
            'ci_low': max(0.0, p - margin) * 100,
            'ci_high': min(1.0, p + margin) * 100,
            'n': n,
        }

    return {
        'sequences': len(repeats),
        'divergent_sequences': len(divergent),
        'token_decisions': totals['upos'],
        'upos': rate('upos'),
        'head': rate('head'),
        'deprel': rate('deprel'),
        'las': rate('las'),
        'upos_kappa': cohens_kappa(upos_pairs),
        'divergent': divergent,
    }


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """Return Cohen's kappa for a list of paired categorical decisions."""
    if not pairs:
        return float('nan')
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum((first[c] / n) * (second[c] / n) for c in set(first) | set(second))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def find_near_duplicate_clusters(
    items: list[tuple[str, conllu.TokenList]] | None = None,
    max_edits: int = 2,
    min_similarity: float = 0.7,
) -> list[list[tuple[str, Any]]]:
    """Group sentences into clusters of near-identical word sequences."""
    items = items if items is not None else annotated_sentences()

    by_length: dict[int, list[tuple[str, Any, tuple[str, ...]]]] = defaultdict(list)
    for session, sentence in items:
        key = _key(sentence)
        if len(key) >= MIN_SEQUENCE_LENGTH:
            by_length[len(key)].append((session, sentence, key))

    clusters: list[list[tuple[str, Any]]] = []
    for group in by_length.values():
        representatives: list[tuple[tuple[str, ...], list[tuple[str, Any]]]] = []
        for session, sentence, key in group:
            for rep_key, members in representatives:
                distance = sum(x != y for x, y in zip(rep_key, key, strict=True))
                similarity = 1 - distance / len(key)
                if distance <= max_edits and similarity >= min_similarity:
                    members.append((session, sentence))
                    break
            else:
                representatives.append((key, [(session, sentence)]))

        clusters += [members for _, members in representatives if len(members) > 1 and len({s for s, _ in members}) > 1]
    return clusters


def score_near_duplicate_clusters(clusters: list[list[tuple[str, Any]]]) -> dict[str, Any]:
    """Measure agreement on the shared tokens of near-duplicate clusters."""
    totals: Counter[str] = Counter()
    agreed: Counter[str] = Counter()
    upos_pairs: list[tuple[str, str]] = []

    for members in clusters:
        reference = members[0][1]
        for _, other in members[1:]:
            for a, b in zip(words(reference), words(other), strict=True):
                if a['form'].lower() != b['form'].lower():
                    continue  # it's a substituted position, so there's fuck all to compare
                for field in ('upos', 'head', 'deprel'):
                    totals[field] += 1
                    agreed[field] += a[field] == b[field]
                totals['las'] += 1
                agreed['las'] += a['head'] == b['head'] and a['deprel'] == b['deprel']
                upos_pairs.append((a['upos'], b['upos']))

    def rate(field: str) -> float:
        return agreed[field] / totals[field] * 100 if totals[field] else float('nan')

    return {
        'clusters': len(clusters),
        'comparisons': sum(len(m) - 1 for m in clusters),
        'token_decisions': totals['upos'],
        'upos': rate('upos'),
        'head': rate('head'),
        'deprel': rate('deprel'),
        'las': rate('las'),
        'upos_kappa': cohens_kappa(upos_pairs),
    }
