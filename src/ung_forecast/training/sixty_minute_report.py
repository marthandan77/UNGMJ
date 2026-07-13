"""Statistical approval and deterministic reporting for the 60-minute horizon."""

from __future__ import annotations

from dataclasses import dataclass

from ung_forecast.training.empirical import BenchmarkMetrics
from ung_forecast.training.sixty_minute_evaluation import SixtyMinuteEvaluationResult
from ung_forecast.validation.approval import (
    ApprovalDecision,
    StatisticalApprovalCriteria,
    evaluate_statistical_approval,
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


def build_sixty_minute_research_report(
    evaluation: SixtyMinuteEvaluationResult,
    *,
    criteria: StatisticalApprovalCriteria | None = None,
) -> SixtyMinuteResearchReport:
    """Apply predeclared statistical gates without economic or trading approval."""

    effective = criteria or StatisticalApprovalCriteria()
    fold_approvals: list[SixtyMinuteFoldApproval] = []
    reasons: list[str] = []

    for fold in evaluation.folds:
        decision = evaluate_statistical_approval(
            fold.metrics.elastic_net,
            fold.metrics.unconditional,
            effective,
        )
        best = fold.metrics.best_brier_name()
        if best != "elastic_net":
            decision = ApprovalDecision(
                approved=False,
                reasons=tuple((*decision.reasons, "elastic_net_not_best_brier_benchmark")),
            )
        fold_approvals.append(
            SixtyMinuteFoldApproval(
                fold_number=fold.fold_number,
                decision=decision,
                best_brier_model=best,
            )
        )

    aggregate_decision = evaluate_statistical_approval(
        evaluation.aggregate.elastic_net,
        evaluation.aggregate.unconditional,
        effective,
    )
    reasons.extend(aggregate_decision.reasons)
    aggregate_best = evaluation.aggregate.best_brier_name()
    if aggregate_best != "elastic_net":
        reasons.append("elastic_net_not_best_brier_benchmark")
    if not all(item.decision.approved for item in fold_approvals):
        reasons.append("one_or_more_walk_forward_folds_failed")

    unique_reasons = tuple(dict.fromkeys(reasons))
    statistically_approved = aggregate_decision.approved and not unique_reasons
    return SixtyMinuteResearchReport(
        horizon_key="60m",
        fold_count=len(evaluation.folds),
        statistically_approved=statistically_approved,
        approval_reasons=unique_reasons,
        aggregate_best_brier_model=aggregate_best,
        aggregate_metrics=evaluation.aggregate,
        folds=tuple(fold_approvals),
    )
