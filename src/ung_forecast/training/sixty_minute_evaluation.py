"""Fold-level model evaluation for the 60-minute research horizon."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ung_forecast.models.calibration import CalibrationConfig, MulticlassProbabilityCalibrator
from ung_forecast.models.elastic_net import ElasticNetConfig, ElasticNetMultinomialModel
from ung_forecast.models.plain_logistic import PlainLogisticConfig, PlainMultinomialLogisticModel
from ung_forecast.schemas import ProbabilityForecast
from ung_forecast.training.empirical import BenchmarkMetrics
from ung_forecast.training.sixty_minute_runner import SixtyMinuteRunPlan
from ung_forecast.validation.approval import ValidationMetrics
from ung_forecast.validation.baselines import (
    RecencyWeightedBaselineConfig,
    constant_probability_frame,
    recency_weighted_class_probabilities,
)
from ung_forecast.validation.metrics import (
    CLASS_ORDER,
    class_frequency_baseline,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
)


@dataclass(frozen=True, slots=True)
class SixtyMinuteEvaluationConfig:
    calibration_method: str = "platt"
    recency_half_life: float = 250.0
    elastic_net: ElasticNetConfig | None = None
    plain_logistic: PlainLogisticConfig | None = None

    def __post_init__(self) -> None:
        if self.recency_half_life <= 0:
            raise ValueError("recency_half_life must be positive")


@dataclass(frozen=True, slots=True)
class SixtyMinuteFoldEvaluation:
    fold_number: int
    selected_lower_multiplier: float
    selected_upper_multiplier: float
    metrics: BenchmarkMetrics


@dataclass(frozen=True, slots=True)
class SixtyMinuteEvaluationResult:
    folds: tuple[SixtyMinuteFoldEvaluation, ...]
    aggregate: BenchmarkMetrics


def _probability_frame(
    forecasts: list[ProbabilityForecast],
    index: pd.Index,
) -> pd.DataFrame:
    rows = [
        {
            "LOWER_FIRST": item.lower_first,
            "UPPER_FIRST": item.upper_first,
            "NEITHER": item.neither,
        }
        for item in forecasts
    ]
    return pd.DataFrame(rows, index=index, columns=CLASS_ORDER)


def _metrics(target: pd.Series, probabilities: pd.DataFrame) -> ValidationMetrics:
    return ValidationMetrics(
        brier_score=multiclass_brier_score(target, probabilities),
        log_loss=multiclass_log_loss(target, probabilities),
        calibration_error=expected_calibration_error(target, probabilities),
        economic_value=float("nan"),
        sample_count=len(target),
    )


def _calibrated_probabilities(
    validation_raw: pd.DataFrame,
    validation_target: pd.Series,
    test_raw: pd.DataFrame,
    *,
    method: str,
) -> pd.DataFrame:
    calibrator = MulticlassProbabilityCalibrator(CalibrationConfig(method=method))
    calibrator.fit(validation_raw, validation_target)
    return calibrator.transform(test_raw)


def evaluate_sixty_minute_plan(
    plan: SixtyMinuteRunPlan,
    *,
    config: SixtyMinuteEvaluationConfig | None = None,
) -> SixtyMinuteEvaluationResult:
    """Train on each planned fold and evaluate only its untouched test rows."""

    effective = config or SixtyMinuteEvaluationConfig()
    fold_results: list[SixtyMinuteFoldEvaluation] = []
    aggregate_targets: list[pd.Series] = []
    aggregate_probabilities: dict[str, list[pd.DataFrame]] = {
        "unconditional": [],
        "recency_weighted": [],
        "plain_logistic_raw": [],
        "plain_logistic": [],
        "elastic_net_raw": [],
        "elastic_net": [],
    }

    for fold in plan.folds:
        dataset = fold.dataset
        train_x = dataset.features.loc[fold.training_index]
        train_y = dataset.target.loc[fold.training_index]
        validation_x = dataset.features.loc[fold.validation_index]
        validation_y = dataset.target.loc[fold.validation_index]
        test_x = dataset.features.loc[fold.test_index]
        test_y = dataset.target.loc[fold.test_index]

        elastic = ElasticNetMultinomialModel(effective.elastic_net)
        elastic.fit(train_x, train_y)
        elastic_validation_raw = _probability_frame(
            elastic.predict_probabilities(validation_x), validation_x.index
        )
        elastic_test_raw = _probability_frame(elastic.predict_probabilities(test_x), test_x.index)
        elastic_test = _calibrated_probabilities(
            elastic_validation_raw,
            validation_y,
            elastic_test_raw,
            method=effective.calibration_method,
        )

        plain = PlainMultinomialLogisticModel(effective.plain_logistic)
        plain.fit(train_x, train_y)
        plain_validation_raw = _probability_frame(
            plain.predict_probabilities(validation_x), validation_x.index
        )
        plain_test_raw = _probability_frame(plain.predict_probabilities(test_x), test_x.index)
        plain_test = _calibrated_probabilities(
            plain_validation_raw,
            validation_y,
            plain_test_raw,
            method=effective.calibration_method,
        )

        unconditional_vector = class_frequency_baseline(train_y).iloc[0]
        unconditional = constant_probability_frame(unconditional_vector, test_x.index)
        recency_vector = recency_weighted_class_probabilities(
            train_y,
            config=RecencyWeightedBaselineConfig(half_life=effective.recency_half_life),
        )
        recency = constant_probability_frame(recency_vector, test_x.index)

        metrics = BenchmarkMetrics(
            unconditional=_metrics(test_y, unconditional),
            recency_weighted=_metrics(test_y, recency),
            plain_logistic_raw=_metrics(test_y, plain_test_raw),
            plain_logistic=_metrics(test_y, plain_test),
            elastic_net_raw=_metrics(test_y, elastic_test_raw),
            elastic_net=_metrics(test_y, elastic_test),
        )
        fold_results.append(
            SixtyMinuteFoldEvaluation(
                fold_number=fold.fold_number,
                selected_lower_multiplier=fold.selected_barrier.lower_multiplier,
                selected_upper_multiplier=fold.selected_barrier.upper_multiplier,
                metrics=metrics,
            )
        )
        aggregate_targets.append(test_y)
        aggregate_probabilities["unconditional"].append(unconditional)
        aggregate_probabilities["recency_weighted"].append(recency)
        aggregate_probabilities["plain_logistic_raw"].append(plain_test_raw)
        aggregate_probabilities["plain_logistic"].append(plain_test)
        aggregate_probabilities["elastic_net_raw"].append(elastic_test_raw)
        aggregate_probabilities["elastic_net"].append(elastic_test)

    if not fold_results:
        raise ValueError("No 60-minute folds were available for evaluation")

    target = pd.concat(aggregate_targets)
    combined = {name: pd.concat(frames) for name, frames in aggregate_probabilities.items()}
    aggregate = BenchmarkMetrics(
        unconditional=_metrics(target, combined["unconditional"]),
        recency_weighted=_metrics(target, combined["recency_weighted"]),
        plain_logistic_raw=_metrics(target, combined["plain_logistic_raw"]),
        plain_logistic=_metrics(target, combined["plain_logistic"]),
        elastic_net_raw=_metrics(target, combined["elastic_net_raw"]),
        elastic_net=_metrics(target, combined["elastic_net"]),
    )
    return SixtyMinuteEvaluationResult(folds=tuple(fold_results), aggregate=aggregate)
