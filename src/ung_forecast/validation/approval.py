"""Predeclared statistical approval gate for horizon models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    brier_score: float
    log_loss: float
    calibration_error: float
    economic_value: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class ApprovalCriteria:
    minimum_samples: int = 250
    maximum_calibration_error: float = 0.10
    minimum_brier_improvement: float = 0.0
    minimum_log_loss_improvement: float = 0.0
    minimum_economic_value: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        if not 0.0 <= self.maximum_calibration_error <= 1.0:
            raise ValueError("maximum_calibration_error must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    approved: bool
    reasons: tuple[str, ...]


def evaluate_approval(
    model: ValidationMetrics,
    baseline: ValidationMetrics,
    criteria: ApprovalCriteria,
) -> ApprovalDecision:
    reasons: list[str] = []
    if model.sample_count < criteria.minimum_samples:
        reasons.append("insufficient_samples")
    if model.calibration_error > criteria.maximum_calibration_error:
        reasons.append("calibration_error_too_high")
    if baseline.brier_score - model.brier_score <= criteria.minimum_brier_improvement:
        reasons.append("brier_not_better_than_baseline")
    if baseline.log_loss - model.log_loss <= criteria.minimum_log_loss_improvement:
        reasons.append("log_loss_not_better_than_baseline")
    if model.economic_value <= criteria.minimum_economic_value:
        reasons.append("economic_value_not_positive")
    return ApprovalDecision(approved=not reasons, reasons=tuple(reasons))
