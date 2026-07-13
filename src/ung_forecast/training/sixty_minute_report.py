"""Statistical approval and deterministic reporting for the 60-minute horizon."""

from __future__ import annotations

from dataclasses import dataclass

from ung_forecast.training.empirical import BenchmarkMetrics
from ung_forecast.training.sixty_minute_evaluation import SixtyMinuteEvaluationResult
from ung_forecast.validation.approval import (
    ApprovalDecision,
    StatisticalApprovalCriteria,
    ValidationMetrics,
)


@dataclass(frozen=True, slots=True)
class SixtyMinuteFoldApproval:
    fold_number: int
    decision: ApprovalDecision
    best_brier_model: str


@dataclass(frozen=True, slots=True)
class SixtyMinuteResearchReport:
    horizon_key: str
    fold_count: int
    statistically_approved: bool
    approval_reasons: tuple[str, ...]
    aggregate_best_brier_model: str
    aggregate_metrics: BenchmarkMetrics
    folds: tuple[SixtyMinuteFoldApproval, ...]


def _external_benchmarks(metrics: BenchmarkMetrics) -> dict[str, ValidationMetrics]:
    """Return independent challengers, excluding Elastic-Net transform diagnostics."""

    return {
        "unconditional": metrics.unconditional,
        "recency_weighted": metrics.recency_weighted,
        "plain_logistic": metrics.plain_logistic,
    }


def _best_production_or_external_brier_name(metrics: BenchmarkMetrics) -> str:
    values = {
        **{name: item.brier_score for name, item in _external_benchmarks(metrics).items()},
        "elastic_net": metrics.elastic_net.brier_score,
    }
    return min(values, key=values.__getitem__)


def _evaluate_production_stream(
    metrics: BenchmarkMetrics,
    criteria: StatisticalApprovalCriteria,
) -> ApprovalDecision:
    """Require the frozen production stream to beat every external challenger."""

    model = metrics.elastic_net
    reasons: list[str] = []
    if model.sample_count < criteria.minimum_samples:
        reasons.append("insufficient_samples")
    if model.calibration_error > criteria.maximum_calibration_error:
        reasons.append("calibration_error_too_high")

    for name, benchmark in _external_benchmarks(metrics).items():
        if benchmark.brier_score - model.brier_score <= criteria.minimum_brier_improvement:
            reasons.append(f"brier_not_better_than_{name}")
        if benchmark.log_loss - model.log_loss <= criteria.minimum_log_loss_improvement:
            reasons.append(f"log_loss_not_better_than_{name}")

    return ApprovalDecision(
        approved=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def build_sixty_minute_research_report(
    evaluation: SixtyMinuteEvaluationResult,
    *,
    criteria: StatisticalApprovalCriteria | None = None,
) -> SixtyMinuteResearchReport:
    """Apply predeclared statistical gates without economic or trading approval."""

    effective = criteria or StatisticalApprovalCriteria()
    fold_approvals: list[SixtyMinuteFoldApproval] = []

    for fold in evaluation.folds:
        decision = _evaluate_production_stream(fold.metrics, effective)
        fold_approvals.append(
            SixtyMinuteFoldApproval(
                fold_number=fold.fold_number,
                decision=decision,
                best_brier_model=_best_production_or_external_brier_name(fold.metrics),
            )
        )

    aggregate_decision = _evaluate_production_stream(evaluation.aggregate, effective)
    reasons = list(aggregate_decision.reasons)
    if not all(item.decision.approved for item in fold_approvals):
        reasons.append("one_or_more_walk_forward_folds_failed")

    unique_reasons = tuple(dict.fromkeys(reasons))
    statistically_approved = aggregate_decision.approved and not unique_reasons
    return SixtyMinuteResearchReport(
        horizon_key="60m",
        fold_count=len(evaluation.folds),
        statistically_approved=statistically_approved,
        approval_reasons=unique_reasons,
        aggregate_best_brier_model=_best_production_or_external_brier_name(evaluation.aggregate),
        aggregate_metrics=evaluation.aggregate,
        folds=tuple(fold_approvals),
    )
