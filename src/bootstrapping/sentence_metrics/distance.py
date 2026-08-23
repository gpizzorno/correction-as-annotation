"""Module to calculate the average dependency distance of a sentence."""

from __future__ import annotations

import conllu
import numpy as np


def dep_distance(sentence: conllu.models.TokenList) -> dict[str, float]:
    """Calculate dependency distance metrics for a sentence.

    Arguments:
        sentence: A conllu TokenList representing the sentence.

    Returns:
        A dictionary with average, median, max, min, and standard deviation of dependency distances.

    """
    distances = [
        abs(token['id'] - token['head']) for token in sentence if isinstance(token['id'], int) and token['head'] != 0
    ]

    return {
        'average': sum(distances) / len(distances) if distances else 0.0,
        'median': float(np.median(distances)) if distances else 0.0,
        'max': max(distances) if distances else 0.0,
        'min': min(distances) if distances else 0.0,
        'std_dev': float(np.std(distances)) if distances else 0.0,
    }
