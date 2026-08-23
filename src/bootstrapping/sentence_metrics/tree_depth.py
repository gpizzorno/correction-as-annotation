"""Module to calculate the depth of a dependency tree."""

from __future__ import annotations

import conllu


def _get_branch_depth(node: conllu.models.TokenTree, current_depth: int) -> int:
    """Recursively calculate the depth of a branch in the dependency tree."""
    if not node or not node.children:
        return current_depth
    return max(_get_branch_depth(child, current_depth + 1) for child in node.children)


def tree_depth(sentence: conllu.models.TokenList | conllu.models.TokenTree) -> int:
    """Calculate the depth of a dependency tree.

    Arguments:
        sentence: A conllu TokenList or TokenTree representing the sentence.

    Returns:
        The depth of the dependency tree as an integer.

    """
    if not sentence:
        msg = 'No sentence provided.'
        raise ValueError(msg)

    if not isinstance(sentence, conllu.models.TokenTree):
        if isinstance(sentence, conllu.models.TokenList):
            sentence = sentence.to_tree()
        else:
            msg = 'Input must be a TokenList or TokenTree.'
            raise TypeError(msg)

    # recursively find the depth of the tree
    return _get_branch_depth(sentence, 0)
