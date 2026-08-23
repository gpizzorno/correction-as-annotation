"""Seed selection for the bootstrapping loop."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

import conllu

LENGTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ('short', 1, 6),
    ('medium', 7, 12),
    ('long', 13, 10_000),
)

# sentences drawn per bucket per batch, proportional to the corpus.
DEFAULT_QUOTA: dict[str, int] = {'short': 136, 'medium': 46, 'long': 18}


def word_count(sentence: conllu.TokenList) -> int:
    """Return the number of non-punctuation word tokens in a sentence."""
    return sum(1 for token in sentence if isinstance(token['id'], int) and token['upos'] != 'PUNCT')


def bucket_of(sentence: conllu.TokenList) -> str:
    """Return the length-bucket label for a sentence."""
    length = word_count(sentence)
    for label, low, high in LENGTH_BUCKETS:
        if low <= length <= high:
            return label
    msg = f'sentence length {length} falls outside every bucket'
    raise ValueError(msg)


def bucket_corpus(sentences: Iterable[conllu.TokenList]) -> dict[str, list[str]]:
    """Group sentence IDs by length bucket."""
    buckets: dict[str, list[str]] = {label: [] for label, _, _ in LENGTH_BUCKETS}
    for sentence in sentences:
        buckets[bucket_of(sentence)].append(sentence.metadata['sent_id'])
    return buckets


def select_seed(
    sentences: Sequence[conllu.TokenList],
    exclude: Iterable[str] = (),
    quota: dict[str, int] | None = None,
    seed: int | None = None,
) -> list[str]:
    """Draw one seed batch of sentence IDs.

    Arguments:
        sentences: the full corpus.
        exclude: sentence IDs already used by earlier batches.
        quota: sentences to draw per bucket. Defaults to 'DEFAULT_QUOTA'.
        seed: RNG seed.

    Returns:
        Sentence IDs for the new batch.

    Raises:
        ValueError: if a bucket cannot satisfy its quota after exclusions.

    """
    quota = quota or DEFAULT_QUOTA
    used = set(exclude)
    rng = random.Random(seed)  # nosec: corpus sampling, not security-sensitive

    buckets = bucket_corpus(sentences)
    selected: list[str] = []
    for label, wanted in quota.items():
        available = [sent_id for sent_id in buckets[label] if sent_id not in used]
        if len(available) < wanted:
            msg = f'bucket {label!r} has {len(available)} unused sentences but {wanted} were requested'
            raise ValueError(msg)
        selected.extend(rng.sample(available, wanted))
    return selected


def extract_sentences(
    sentences: Iterable[conllu.TokenList],
    sent_ids: Iterable[str],
) -> list[conllu.TokenList]:
    """Return the sentences whose IDs are in 'sent_ids', in corpus order."""
    wanted = set(sent_ids)
    return [s for s in sentences if s.metadata.get('sent_id') in wanted]


def describe_batch(sentences: Sequence[conllu.TokenList]) -> dict[str, int]:
    """Return the bucket distribution of a batch, for checking against the quota."""
    counts = dict.fromkeys((label for label, _, _ in LENGTH_BUCKETS), 0)
    for sentence in sentences:
        counts[bucket_of(sentence)] += 1
    return counts
