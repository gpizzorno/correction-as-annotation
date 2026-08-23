"""Module to calculate the grammatical complexity score of a sentence."""

from __future__ import annotations

import conllu

from .token_count import token_count

DEFAULT_WEIGHTS = {
    'non_core_dependents': 0.5,
    'complex_clauses': 0.75,
    'special_relationships': 1.0,
}

DEFAULT_TAGSETS = {
    'non_core_dependents': [
        'advmod',
        'appos',
        'aux',
        'case',
        'cc',
        'compound',
        'det',
        'discourse',
        'dislocated',
        'expl',
        'fixed',
        'flat',
        'mark',
        'nmod:poss',
        'nummod',
        'obl',
        'ref',
        'vocative',
    ],
    'complex_clauses': [
        'acl',
        'advcl',
        'ccomp',
        'conj',
        'cc',
        'csubj',
        'csubj:pass',
        'mark',
        'xcomp',
    ],
    'special_relationships': [
        'list',
        'orphan',
        'parataxis',
        'dislocated',
        'parenthetical',
    ],
}


def gbsc_score(
    sentence: conllu.models.TokenList,
    normalize: bool = False,  # noqa: FBT001
    exclude_fw: bool = False,  # noqa: FBT001
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    tagsets: dict[str, list[str]] = DEFAULT_TAGSETS,
) -> float:
    """Calculate the GBSC score for a sentence.

    GBSC stands for either 'Gabe Bullshit Sentence Complexity' or 'Grammatical-Based Sentence Complexity', your pick.

    Arguments:
        sentence: A conllu TokenList representing the sentence to analyze.
        normalize: If True, normalize the score by the token count.
        exclude_fw: If True, exclude function words when normalizing.
        weights: A dictionary containing weights for each category.
        tagsets: A dictionary containing lists of dependency relation tags for each category.

    Returns:
        A float representing the GBSC score.

    """
    # weights:
    w_ncd = weights.get('non_core_dependents', DEFAULT_WEIGHTS['non_core_dependents'])
    w_ccl = weights.get('complex_clauses', DEFAULT_WEIGHTS['complex_clauses'])
    w_spr = weights.get('special_relationships', DEFAULT_WEIGHTS['special_relationships'])
    ncd, ccl, spr = 0, 0, 0
    for token in sentence:
        dp = token['deprel']
        if dp in tagsets.get('non_core_dependents', DEFAULT_TAGSETS['non_core_dependents']):
            ncd += 1
        elif dp in tagsets.get('complex_clauses', DEFAULT_TAGSETS['complex_clauses']):
            ccl += 1
        elif dp in tagsets.get('special_relationships', DEFAULT_TAGSETS['special_relationships']):
            spr += 1

    score = (ncd * w_ncd) + (ccl * w_ccl) + (spr * w_spr)

    if normalize:
        length = token_count(sentence, exclude_fw)
        score = score / length if length > 0 else 0

    return score
