"""Module for calculating token count in sentences."""

from __future__ import annotations

import conllu

FUNCTION_WORD_UPOS = {
    'NOUN',
    'VERB',
    'ADJ',
    'ADV',
}


def token_count(sentence: conllu.models.TokenList, exclude_fw: bool = False) -> int:  # noqa: FBT001
    """Calculate the number of tokens in a sentence.

    It only counts non-punctuation tokens.

    Arguments:
        sentence: A conllu TokenList representing the sentence.
        exclude_fw: If True, exclude function words (determiners, auxiliaries, etc.).

    Returns:
        The number of tokens in the sentence.

    """
    if exclude_fw:
        np_tokens = [t for t in sentence if t.get('upos') != 'PUNCT' and t.get('upos') in FUNCTION_WORD_UPOS]
    else:
        np_tokens = [t for t in sentence if t.get('upos') != 'PUNCT']

    return len(np_tokens)
