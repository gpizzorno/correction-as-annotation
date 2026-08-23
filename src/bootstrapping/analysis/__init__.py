"""Interface for the analysis module."""

from __future__ import annotations

from .comparison import (
    align_sentences,
    align_tokens,
    alignment_report,
    bootstrap_test,
    classify_construction,
    compare_predictions,
    compute_construction_accuracy,
    compute_training_stats,
    get_error_example,
    head_projection,
    paired_t_test,
    sentence_score,
    track_error_resolution,
)

__all__ = [
    'align_sentences',
    'align_tokens',
    'alignment_report',
    'bootstrap_test',
    'classify_construction',
    'compare_predictions',
    'compute_construction_accuracy',
    'compute_training_stats',
    'get_error_example',
    'head_projection',
    'paired_t_test',
    'sentence_score',
    'track_error_resolution',
]
