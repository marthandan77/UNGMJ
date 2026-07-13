from __future__ import annotations

from ung_forecast.models.elastic_net import ElasticNetConfig
from ung_forecast.training.empirical import BenchmarkMetrics
from ung_forecast.training.fold_diagnostics import (
    CoefficientDiagnostics,
    FeatureDriftDiagnostics,
)
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


def _benchmarks(
    elastic_brier: float,
    *,
    elastic_raw_brier: float | None = None,
    plain_brier: float = 0.25,
) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        unconditional=_metrics(0.40, 0.9),
        recency_weighted=_metrics(0.38, 0.8),
        plain_logistic_raw=_metrics(plain_brier + 0.02, 0.65),
        plain_logistic=_metrics(plain_brier, 0.60),
        elastic_net_raw=_metrics(
            elastic_raw_brier if elastic_raw_brier is not None else elastic_brier + 0.02,
            0.55,
        ),
        elastic_net=_metrics(elastic_brier, 0.50),
    )


def _fold(
    number: int,
    lower: float,
    upper: float,
    benchmarks: BenchmarkMetrics,
) -> SixtyMinuteFoldEvaluation:
    counts = ClassCounts(lower_first=100, upper_first=100, neither=100)
    coefficients = CoefficientDiagnostics(l2_norm=1.0, nonzero_count=3, total_count=6)
    return SixtyMinuteFoldEvaluation(
        fold_number=number,
        selected_lower_multiplier=lower,
        selected_upper_multiplier=upper,
        metrics=benchmarks,
        training_class_counts=counts,
        validation_class_counts=counts,
        test_class_counts=counts,
        plain_probability_mode="calibrated",
        elastic_probability_mode="calibrated",
        calibration_fit_rows=150,
        calibration_selection_rows=150,
        plain_selection_raw_brier=benchmarks.plain_logistic_raw.brier_score + 0.01,
        plain_selection_calibrated_brier=benchmarks.plain_logistic.brier_score,
        elastic_selection_raw_brier=benchmarks.elastic_net_raw.brier_score + 0.01,
        elastic_selection_calibrated_brier=benchmarks.elastic_net.brier_score,
        plain_calibrated_test_brier=benchmarks.plain_logistic.brier_score,
        elastic_calibrated_test_brier=benchmarks.elastic_net.brier_score,
        plain_calibration_brier_delta=(
            benchmarks.plain_logistic.brier_score - benchmarks.plain_logistic_raw.brier_score
        ),
        elastic_calibration_brier_delta=(
            benchmarks.elastic_net.brier_score - benchmarks.elastic_net_raw.brier_score
        ),
        best_brier_model=benchmarks.best_brier_name(),
        elastic_net_config=ElasticNetConfig(),
        elastic_net_selection=None,
        feature_drift=FeatureDriftDiagnostics(
            validation_mahalanobis=0.0,
            test_mahalanobis=0.0,
        ),
        elastic_coefficients=coefficients,
        plain_coefficients=coefficients,
    )


def test_approves_only_when_production_elastic_net_wins_every_external_gate() -> None:
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


def test_internal_raw_elastic_net_is_diagnostic_not_external_challenger() -> None:
    benchmarks = _benchmarks(0.20, elastic_raw_brier=0.15)
    evaluation = SixtyMinuteEvaluationResult(
        folds=(_fold(1, 1.0, 1.0, benchmarks),),
        aggregate=benchmarks,
    )
    report = build_sixty_minute_research_report(evaluation)
    assert report.statistically_approved is True
    assert report.approval_reasons == ()
    assert report.aggregate_best_brier_model == "elastic_net"


def test_rejects_when_plain_logistic_production_stream_has_better_brier() -> None:
    benchmarks = _benchmarks(0.30, plain_brier=0.20)
    evaluation = SixtyMinuteEvaluationResult(
        folds=(_fold(1, 1.0, 1.0, benchmarks),),
        aggregate=benchmarks,
    )
    report = build_sixty_minute_research_report(evaluation)
    assert report.statistically_approved is False
    assert "brier_not_better_than_plain_logistic" in report.approval_reasons
    assert report.aggregate_best_brier_model == "plain_logistic"


def test_accepts_small_fold_misses_when_win_rate_and_regret_are_stable() -> None:
    winning = _benchmarks(0.20)
    small_miss = _benchmarks(0.255, plain_brier=0.25)
    folds = tuple(
        _fold(number, 1.0, 1.0, small_miss if number in (2, 4) else winning)
        for number in range(1, 9)
    )
    evaluation = SixtyMinuteEvaluationResult(folds=folds, aggregate=winning)
    report = build_sixty_minute_research_report(evaluation)
    assert report.statistically_approved is True
    assert report.approval_reasons == ()
    assert report.folds[1].decision.approved is False
    assert report.folds[3].decision.approved is False


def test_rejects_material_fold_instability() -> None:
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
    assert "fold_win_rate_below_plain_logistic" in report.approval_reasons
    assert "fold_brier_regret_too_high_plain_logistic" in report.approval_reasons
