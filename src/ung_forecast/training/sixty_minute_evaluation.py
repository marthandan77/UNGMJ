"""Fold-level model evaluation for the 60-minute research horizon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ung_forecast.models.calibration import CalibrationConfig, MulticlassProbabilityCalibrator
from ung_forecast.models.elastic_net import ElasticNetConfig, ElasticNetMultinomialModel
from ung_forecast.models.plain_logistic import PlainLogisticConfig, PlainMultinomialLogisticModel
from ung_forecast.schemas import ProbabilityForecast
from ung_forecast.training.empirical import BenchmarkMetrics
from ung_forecast.training.runner_60m import SixtyMinuteRunPlan
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

ProbabilityMode = Literal["raw", "calibrated"]


@dataclass(frozen=True, slots=True)
class SixtyMinuteEvaluationConfig:
    calibration_method: str = "platt"
    calibration_fit_fraction: float = 0.5
    minimum_calibration_fit_rows: int = 10
    minimum_calibration_selection_rows: int = 10
    recency_half_life: float = 250.0
    elastic_net: ElasticNetConfig | None = None
    plain_logistic: PlainLogisticConfig | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.calibration_fit_fraction < 1.0:
            raise ValueError("calibration_fit_fraction must be strictly between zero and one")
        if self.minimum_calibration_fit_rows <= 0:
            raise ValueError("minimum_calibration_fit_rows must be positive")
        if self.minimum_calibration_selection_rows <= 0:
            raise ValueError("minimum_calibration_selection_rows must be positive")
        if self.recency_half_life <= 0:
            raise ValueError("recency_half_life must be positive")


@dataclass(frozen=True, slots=True)
class ClassCounts:
    lower_first: int
    upper_first: int
    neither: int

    @property
    def total(self) -> int:
        return self.lower_first + self.upper_first + self.neither


@dataclass(frozen=True, slots=True)
class CalibrationSelection:
    selected_probabilities: pd.DataFrame
    calibrated_probabilities: pd.DataFrame
    selected_mode: ProbabilityMode
    fit_rows: int
    selection_rows: int
    selection_raw_brier: float
    selection_calibrated_brier: float


@dataclass(frozen=True, slots=True)
class SixtyMinuteFoldEvaluation:
    fold_number: int
    selected_lower_multiplier: float
    selected_upper_multiplier: float
    metrics: BenchmarkMetrics
    training_class_counts: ClassCounts
    validation_class_counts: ClassCounts
    test_class_counts: ClassCounts
    plain_probability_mode: ProbabilityMode
    elastic_probability_mode: ProbabilityMode
    calibration_fit_rows: int
    calibration_selection_rows: int
    plain_selection_raw_brier: float
    plain_selection_calibrated_brier: float
    elastic_selection_raw_brier: float
    elastic_selection_calibrated_brier: float
    plain_calibrated_test_brier: float
    elastic_calibrated_test_brier: float
    plain_calibration_brier_delta: float
    elastic_calibration_brier_delta: float
    best_brier_model: str


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


def _class_counts(target: pd.Series) -> ClassCounts:
    counts = target.astype(str).value_counts()
    return ClassCounts(
        lower_first=int(counts.get("LOWER_FIRST", 0)),
        upper_first=int(counts.get("UPPER_FIRST", 0)),
        neither=int(counts.get("NEITHER", 0)),
    )


def _split_validation_index(
    index: pd.Index,
    *,
    fit_fraction: float,
    minimum_fit_rows: int,
    minimum_selection_rows: int,
) -> tuple[pd.Index, pd.Index]:
    if len(index) < minimum_fit_rows + minimum_selection_rows:
        raise ValueError("Validation interval is too short for calibration fit and selection")
    proposed_fit_rows = int(len(index) * fit_fraction)
    fit_rows = max(minimum_fit_rows, proposed_fit_rows)
    fit_rows = min(fit_rows, len(index) - minimum_selection_rows)
    fit_index = index[:fit_rows]
    selection_index = index[fit_rows:]
    if len(fit_index) < minimum_fit_rows or len(selection_index) < minimum_selection_rows:
        raise ValueError("Calibration split failed minimum row requirements")
    return fit_index, selection_index


def _select_calibration_mode(
    validation_raw: pd.DataFrame,
    validation_target: pd.Series,
    test_raw: pd.DataFrame,
    *,
    method: str,
    fit_fraction: float,
    minimum_fit_rows: int,
    minimum_selection_rows: int,
) -> CalibrationSelection:
    fit_index, selection_index = _split_validation_index(
        validation_raw.index,
        fit_fraction=fit_fraction,
        minimum_fit_rows=minimum_fit_rows,
        minimum_selection_rows=minimum_selection_rows,
    )
    calibrator = MulticlassProbabilityCalibrator(CalibrationConfig(method=method))
    calibrator.fit(validation_raw.loc[fit_index], validation_target.loc[fit_index])
    selection_calibrated = calibrator.transform(validation_raw.loc[selection_index])
    selection_raw = validation_raw.loc[selection_index]
    selection_target = validation_target.loc[selection_index]
    raw_brier = multiclass_brier_score(selection_target, selection_raw)
    calibrated_brier = multiclass_brier_score(selection_target, selection_calibrated)
    calibrated_test = calibrator.transform(test_raw)
    selected_mode: ProbabilityMode = "calibrated" if calibrated_brier < raw_brier else "raw"
    selected_test = calibrated_test if selected_mode == "calibrated" else test_raw
    return CalibrationSelection(
        selected_probabilities=selected_test,
        calibrated_probabilities=calibrated_test,
        selected_mode=selected_mode,
        fit_rows=len(fit_index),
        selection_rows=len(selection_index),
        selection_raw_brier=raw_brier,
        selection_calibrated_brier=calibrated_brier,
    )


def evaluate_sixty_minute_plan(
    plan: SixtyMinuteRunPlan,
    *,
    config: SixtyMinuteEvaluationConfig | None = None,
) -> SixtyMinuteEvaluationResult:
    """Train each fold and select calibration without inspecting test outcomes."""

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
        elastic_selection = _select_calibration_mode(
            elastic_validation_raw,
            validation_y,
            elastic_test_raw,
            method=effective.calibration_method,
            fit_fraction=effective.calibration_fit_fraction,
            minimum_fit_rows=effective.minimum_calibration_fit_rows,
            minimum_selection_rows=effective.minimum_calibration_selection_rows,
        )

        plain = PlainMultinomialLogisticModel(effective.plain_logistic)
        plain.fit(train_x, train_y)
        plain_validation_raw = _probability_frame(
            plain.predict_probabilities(validation_x), validation_x.index
        )
        plain_test_raw = _probability_frame(plain.predict_probabilities(test_x), test_x.index)
        plain_selection = _select_calibration_mode(
            plain_validation_raw,
            validation_y,
            plain_test_raw,
            method=effective.calibration_method,
            fit_fraction=effective.calibration_fit_fraction,
            minimum_fit_rows=effective.minimum_calibration_fit_rows,
            minimum_selection_rows=effective.minimum_calibration_selection_rows,
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
            plain_logistic=_metrics(test_y, plain_selection.selected_probabilities),
            elastic_net_raw=_metrics(test_y, elastic_test_raw),
            elastic_net=_metrics(test_y, elastic_selection.selected_probabilities),
        )
        plain_calibrated_test_brier = multiclass_brier_score(
            test_y, plain_selection.calibrated_probabilities
        )
        elastic_calibrated_test_brier = multiclass_brier_score(
            test_y, elastic_selection.calibrated_probabilities
        )
        if plain_selection.fit_rows != elastic_selection.fit_rows:
            raise ValueError("Model calibration fit partitions are inconsistent")
        if plain_selection.selection_rows != elastic_selection.selection_rows:
            raise ValueError("Model calibration selection partitions are inconsistent")
        fold_results.append(
            SixtyMinuteFoldEvaluation(
                fold_number=fold.fold_number,
                selected_lower_multiplier=fold.selected_barrier.lower_multiplier,
                selected_upper_multiplier=fold.selected_barrier.upper_multiplier,
                metrics=metrics,
                training_class_counts=_class_counts(train_y),
                validation_class_counts=_class_counts(validation_y),
                test_class_counts=_class_counts(test_y),
                plain_probability_mode=plain_selection.selected_mode,
                elastic_probability_mode=elastic_selection.selected_mode,
                calibration_fit_rows=elastic_selection.fit_rows,
                calibration_selection_rows=elastic_selection.selection_rows,
                plain_selection_raw_brier=plain_selection.selection_raw_brier,
                plain_selection_calibrated_brier=plain_selection.selection_calibrated_brier,
                elastic_selection_raw_brier=elastic_selection.selection_raw_brier,
                elastic_selection_calibrated_brier=elastic_selection.selection_calibrated_brier,
                plain_calibrated_test_brier=plain_calibrated_test_brier,
                elastic_calibrated_test_brier=elastic_calibrated_test_brier,
                plain_calibration_brier_delta=(
                    plain_calibrated_test_brier - metrics.plain_logistic_raw.brier_score
                ),
                elastic_calibration_brier_delta=(
                    elastic_calibrated_test_brier - metrics.elastic_net_raw.brier_score
                ),
                best_brier_model=metrics.best_brier_name(),
            )
        )
        aggregate_targets.append(test_y)
        aggregate_probabilities["unconditional"].append(unconditional)
        aggregate_probabilities["recency_weighted"].append(recency)
        aggregate_probabilities["plain_logistic_raw"].append(plain_test_raw)
        aggregate_probabilities["plain_logistic"].append(plain_selection.selected_probabilities)
        aggregate_probabilities["elastic_net_raw"].append(elastic_test_raw)
        aggregate_probabilities["elastic_net"].append(elastic_selection.selected_probabilities)

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
