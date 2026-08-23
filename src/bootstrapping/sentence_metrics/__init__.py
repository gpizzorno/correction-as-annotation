"""Interface for the sentence_metrics module."""

from __future__ import annotations

from .distance import dep_distance
from .gbsc import gbsc_score
from .token_count import token_count
from .tree_depth import tree_depth

__all__ = [
    'dep_distance',
    'gbsc_score',
    'token_count',
    'tree_depth',
]
