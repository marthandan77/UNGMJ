"""Statistical approval and deterministic reporting for the 60-minute horizon."""

from __future__ import annotations

from dataclasses import dataclass

from ung_forecast.training.empirical import BenchmarkMetrics
from ung_forecast.training.sixty_minute_evaluation import (
    SixtyMinuteEvaluationResult,
    SixtyMinuteFoldEvaluation,
)
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


def _evaluate_walk_forward_stability(
    folds: tuple[SixtyMinuteFoldEvaluation, ...],
    criteria: StatisticalApprovalCriteria,
) -> ApprovalDecision:
    """Require broad fold support without demanding an implausible perfect record.

    The aggregate gate remains strict. Fold stability is evaluated with two
    predeclared controls: a minimum win rate and a cap on the worst Brier regret.
    This rejects broad or material instability while tolerating small sampling noise.
    """

    if not folds:
        return ApprovalDecision(False, ("no_walk_forward_folds",))

    reasons: list[str] = []
    for name in ("unconditional", "recency_weighted", "plain_logistic"):
        wins = 0
        regrets: list[float] = []
        for fold in folds:
            model_brier = fold.metrics.elastic_net.brier_score
            benchmark_brier = _external_benchmarks(fold.metrics)[name].brier_score
            if benchmark_brier - model_brier > criteria.minimum_brier_improvement:
                wins += 1
            regrets.append(model_brier - benchmark_brier)

        win_rate = wins / len(folds)
        maximum_regret = max(regrets)
        if win_rate < criteria.minimum_fold_win_rate:
            reasons.append(f"fold_win_rate_below_{name}")
        if maximum_regret > criteria.maximum_fold_brier_regret:
            reasons.append(f"fold_brier_regret_too_high_{name}")

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
    stability_decision = _evaluate_walk_forward_stability(evaluation.folds, effective)
    reasons = [*aggregate_decision.reasons, *stability_decision.reasons]

    unique_reasons = tuple(dict.fromkeys(reasons))
    statistically_approved = aggregate_decision.approved and stability_decision.approved
    return SixtyMinuteResearchReport(
        horizon_key="60m",
        fold_count=len(evaluation.folds),
        statistically_approved=statistically_approved,
        approval_reasons=unique_reasons,
        aggregate_best_brier_model=_best_production_or_external_brier_name(evaluation.aggregate),
        aggregate_metrics=evaluation.aggregate,
        folds=tuple(fold_approvals),
    )
