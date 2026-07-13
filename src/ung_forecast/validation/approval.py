"""Predeclared statistical and trading approval gates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    brier_score: float
    log_loss: float
    calibration_error: float
    economic_value: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class StatisticalApprovalCriteria:
    minimum_samples: int = 250
    maximum_calibration_error: float = 0.10
    minimum_brier_improvement: float = 0.0
    minimum_log_loss_improvement: float = 0.0
    minimum_fold_win_rate: float = 0.60
    maximum_fold_brier_regret: float = 0.01

    def __post_init__(self) -> None:
        if self.minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")
        if not 0.0 <= self.maximum_calibration_error <= 1.0:
            raise ValueError("maximum_calibration_error must be within [0, 1]")
        if not 0.0 <= self.minimum_fold_win_rate <= 1.0:
            raise ValueError("minimum_fold_win_rate must be within [0, 1]")
        if not isfinite(self.maximum_fold_brier_regret):
            raise ValueError("maximum_fold_brier_regret must be finite")
        if self.maximum_fold_brier_regret < 0.0:
            raise ValueError("maximum_fold_brier_regret must be non-negative")


@dataclass(frozen=True, slots=True)
class ApprovalCriteria(StatisticalApprovalCriteria):
    """Legacy combined gate used by existing research-runner callers."""

    minimum_economic_value: float = 0.0


@dataclass(frozen=True, slots=True)
class TradingApprovalCriteria:
    minimum_economic_value: float = 0.0
    minimum_shadow_samples: int = 100

    def __post_init__(self) -> None:
        if self.minimum_shadow_samples <= 0:
            raise ValueError("minimum_shadow_samples must be positive")


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    approved: bool
    reasons: tuple[str, ...]


def evaluate_statistical_approval(
    model: ValidationMetrics,
    baseline: ValidationMetrics,
    criteria: StatisticalApprovalCriteria,
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
    return ApprovalDecision(approved=not reasons, reasons=tuple(reasons))


def evaluate_trading_approval(
    *,
    statistical_decision: ApprovalDecision,
    economic_value: float,
    shadow_samples: int,
    execution_model_configured: bool,
    criteria: TradingApprovalCriteria,
) -> ApprovalDecision:
    reasons: list[str] = []
    if not statistical_decision.approved:
        reasons.append("statistical_approval_required")
    if not execution_model_configured:
        reasons.append("execution_model_not_configured")
    if shadow_samples < criteria.minimum_shadow_samples:
        reasons.append("insufficient_shadow_samples")
    if not isfinite(economic_value):
        reasons.append("economic_value_unavailable")
    elif economic_value <= criteria.minimum_economic_value:
        reasons.append("economic_value_not_positive")
    return ApprovalDecision(approved=not reasons, reasons=tuple(reasons))


def evaluate_approval(
    model: ValidationMetrics,
    baseline: ValidationMetrics,
    criteria: ApprovalCriteria,
) -> ApprovalDecision:
    """Backward-compatible combined statistical and economic gate."""

    statistical = evaluate_statistical_approval(model, baseline, criteria)
    reasons = list(statistical.reasons)
    if not isfinite(model.economic_value):
        reasons.append("economic_value_unavailable")
    elif model.economic_value <= criteria.minimum_economic_value:
        reasons.append("economic_value_not_positive")
    return ApprovalDecision(approved=not reasons, reasons=tuple(reasons))
