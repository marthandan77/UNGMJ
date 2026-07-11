"""Purged walk-forward validation utilities and approval gates."""

from .approval import (
    ApprovalCriteria,
    ApprovalDecision,
    ValidationMetrics,
    evaluate_approval,
)
from .metrics import (
    class_frequency_baseline,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
)
from .purge import purge_overlapping_training_rows
from .walk_forward import WalkForwardFold, generate_walk_forward_folds

__all__ = [
    "ApprovalCriteria",
    "ApprovalDecision",
    "ValidationMetrics",
    "WalkForwardFold",
    "class_frequency_baseline",
    "evaluate_approval",
    "expected_calibration_error",
    "generate_walk_forward_folds",
    "multiclass_brier_score",
    "multiclass_log_loss",
    "purge_overlapping_training_rows",
]
