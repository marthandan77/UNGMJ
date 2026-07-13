from __future__ import annotations

from ung_forecast.training.empirical import BenchmarkMetrics
from ung_forecast.training.sixty_minute_evaluation import (
    ClassCounts,
    SixtyMinuteEvaluationResult,
    SixtyMinuteFoldEvaluation,
)
from ung_forecast.training.sixty_minute_report import build_sixty_minute_research_report
from ung_forecast.validation.approval import StatisticalApprovalCriteria, ValidationMetrics


def _metrics(brier: float, log_loss: float = 0.5, calibration: float = 0.05) -> ValidationMetrics:
    return ValidationMetrics(
        brier_score=brier,
        log_loss=log_loss,
        calibration_error=calibration,
        economic_value=float("nan"),
        sample_count=300,
    )


def _benchmarks(elastic_brier: float, plain_brier: float = 0.25) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        unconditional=_metrics(0.40, 0.9),
        recency_weighted=_metrics(0.38, 0.8),
        plain_logistic_raw=_metrics(plain_brier + 0.02, 0.65),
        plain_logistic=_metrics(plain_brier, 0.60),
        elastic_net_raw=_metrics(elastic_brier + 0.02, 0.55),
        elastic_net=_metrics(elastic_brier, 0.50),
    )


def _fold(number: int, lower: float, upper: float, benchmarks: BenchmarkMetrics) -> SixtyMinuteFoldEvaluation:
    counts = ClassCounts(lower_first=100, upper_first=100, neither=100)
    return SixtyMinuteFoldEvaluation(
        fold_number=number,
        selected_lower_multiplier=lower,
        selected_upper_multiplier=upper,
        metrics=benchmarks,
        training_class_counts=counts,
        validation_class_counts=counts,
        test_class_counts=counts,
        plain_calibration_brier_delta=(
            benchmarks.plain_logistic.brier_score - benchmarks.plain_logistic_raw.brier_score
        ),
        elastic_calibration_brier_delta=(
            benchmarks.elastic_net.brier_score - benchmarks.elastic_net_raw.brier_score
        ),
        best_brier_model=benchmarks.best_brier_name(),
    )


def test_approves_only_when_elastic_net_wins_every_gate() -> None:
    benchmarks = _benchmarks(0.20)
    evaluation = SixtyMinuteEvaluationResult(
        folds=(_fold(1, 1.0, 1.0, benchmarks),),
        aggregate=benchmarks,
    )
    report = build_sixty_minute_research_report(
        evaluation,
        criteria=StatisticalApprovalCriteria(minimum_samples=250, maximum_calibration_error=0.10),
    )
    assert report.statistically_approved is True
    assert report.approval_reasons == ()
    assert report.aggregate_best_brier_model == "elastic_net"


def test_rejects_when_plain_logistic_has_better_brier() -> None:
    benchmarks = _benchmarks(0.30, plain_brier=0.20)
    evaluation = SixtyMinuteEvaluationResult(
        folds=(_fold(1, 1.0, 1.0, benchmarks),),
        aggregate=benchmarks,
    )
    report = build_sixty_minute_research_report(evaluation)
    assert report.statistically_approved is False
    assert "elastic_net_not_best_brier_benchmark" in report.approval_reasons


def test_rejects_when_any_fold_fails() -> None:
    winning = _benchmarks(0.20)
    losing = _benchmarks(0.31, plain_brier=0.21)
    evaluation = SixtyMinuteEvaluationResult(
        folds=(
            _fold(1, 1.0, 1.0, winning),
            _fold(2, 1.2, 1.2, losing),
        ),
        aggregate=winning,
    )
    report = build_sixty_minute_research_report(evaluation)
    assert report.statistically_approved is False
    assert "one_or_more_walk_forward_folds_failed" in report.approval_reasons
