"""Empirical walk-forward evaluation for one prebuilt horizon dataset.

Statistical approval remains independent. The legacy ``approved`` field is a
combined research gate and cannot pass without an explicit economic evaluator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ung_forecast.models.calibration import CalibrationConfig, MulticlassProbabilityCalibrator
from ung_forecast.models.elastic_net import ElasticNetConfig, ElasticNetMultinomialModel
from ung_forecast.models.plain_logistic import PlainLogisticConfig, PlainMultinomialLogisticModel
from ung_forecast.training.dataset import TrainingDataset
from ung_forecast.validation.approval import (
    ApprovalCriteria,
    ApprovalDecision,
    StatisticalApprovalCriteria,
    ValidationMetrics,
    evaluate_statistical_approval,
)
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
from ung_forecast.validation.purge import purge_overlapping_training_rows
from ung_forecast.validation.walk_forward import generate_walk_forward_folds

EconomicValueEvaluator = Callable[[pd.Series, pd.DataFrame, pd.DataFrame], float]


@dataclass(frozen=True, slots=True)
class EmpiricalRunConfig:
    minimum_train_size: int
    validation_size: int
    test_size: int
    purge_size: int
    embargo_size: int
    calibration_method: str = "platt"
    recency_half_life: float = 250.0

    def __post_init__(self) -> None:
        if self.recency_half_life <= 0:
            raise ValueError("recency_half_life must be positive")


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    unconditional: ValidationMetrics
    recency_weighted: ValidationMetrics
    plain_logistic: ValidationMetrics
    elastic_net: ValidationMetrics

    def best_brier_name(self) -> str:
        values = {
            "unconditional": self.unconditional.brier_score,
            "recency_weighted": self.recency_weighted.brier_score,
            "plain_logistic": self.plain_logistic.brier_score,
            "elastic_net": self.elastic_net.brier_score,
        }
        return min(values, key=values.__getitem__)


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    fold_number: int
    model: ValidationMetrics
    baseline: ValidationMetrics
    statistical_approval: ApprovalDecision
    benchmarks: BenchmarkMetrics


@dataclass(frozen=True, slots=True)
class EmpiricalRunResult:
    folds: tuple[FoldMetrics, ...]
    statistically_approved: bool
    approved: bool
    approval_reasons: tuple[str, ...]
    aggregate_model: ValidationMetrics
    aggregate_baseline: ValidationMetrics
    aggregate_benchmarks: BenchmarkMetrics


def _probability_frame(
    model: ElasticNetMultinomialModel | PlainMultinomialLogisticModel,
    features: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "LOWER_FIRST": probability.lower_first,
            "UPPER_FIRST": probability.upper_first,
            "NEITHER": probability.neither,
        }
        for probability in model.predict_probabilities(features)
    ]
    return pd.DataFrame(rows, index=features.index, columns=CLASS_ORDER)


def _evaluate_metrics(
    target: pd.Series,
    probabilities: pd.DataFrame,
    metadata: pd.DataFrame,
    economic_evaluator: EconomicValueEvaluator | None,
) -> ValidationMetrics:
    economic_value = (
        float(economic_evaluator(target, probabilities, metadata))
        if economic_evaluator is not None
        else float("nan")
    )
    return ValidationMetrics(
        brier_score=multiclass_brier_score(target, probabilities),
        log_loss=multiclass_log_loss(target, probabilities),
        calibration_error=expected_calibration_error(target, probabilities),
        economic_value=economic_value,
        sample_count=len(target),
    )


def _datetime_feature_index(dataset: TrainingDataset) -> pd.DatetimeIndex:
    if not isinstance(dataset.features.index, pd.DatetimeIndex):
        raise ValueError("Empirical evaluation requires a DatetimeIndex")
    return dataset.features.index


def _unconditional_probabilities(target: pd.Series, index: pd.Index) -> pd.DataFrame:
    vector = class_frequency_baseline(target).iloc[0]
    return constant_probability_frame(vector, index)


def run_empirical_evaluation(
    dataset: TrainingDataset,
    *,
    run_config: EmpiricalRunConfig,
    model_config: ElasticNetConfig | None = None,
    plain_logistic_config: PlainLogisticConfig | None = None,
    approval_criteria: StatisticalApprovalCriteria | None = None,
    economic_evaluator: EconomicValueEvaluator | None = None,
) -> EmpiricalRunResult:
    criteria = approval_criteria or StatisticalApprovalCriteria()
    feature_index = _datetime_feature_index(dataset)
    folds = generate_walk_forward_folds(
        sample_count=len(dataset.features),
        minimum_train_size=run_config.minimum_train_size,
        validation_size=run_config.validation_size,
        test_size=run_config.test_size,
        purge_size=run_config.purge_size,
        embargo_size=run_config.embargo_size,
    )

    fold_results: list[FoldMetrics] = []
    aggregate_targets: list[pd.Series] = []
    aggregate_metadata: list[pd.DataFrame] = []
    aggregate_probabilities: dict[str, list[pd.DataFrame]] = {
        "unconditional": [],
        "recency_weighted": [],
        "plain_logistic": [],
        "elastic_net": [],
    }

    for fold_number, fold in enumerate(folds, start=1):
        train_index = feature_index[list(fold.train)]
        validation_index = feature_index[list(fold.validation)]
        test_index = feature_index[list(fold.test)]
        train_index = purge_overlapping_training_rows(
            train_index,
            dataset.label_end_time,
            evaluation_start=validation_index[0],
        )
        if train_index.empty:
            raise ValueError("Purging removed every training observation")

        training_features = dataset.features.loc[train_index]
        training_target = dataset.target.loc[train_index]
        validation_features = dataset.features.loc[validation_index]
        validation_target = dataset.target.loc[validation_index]
        test_features = dataset.features.loc[test_index]
        test_target = dataset.target.loc[test_index]
        test_metadata = dataset.metadata.loc[test_index]

        elastic_net = ElasticNetMultinomialModel(model_config)
        elastic_net.fit(training_features, training_target)
        validation_raw = _probability_frame(elastic_net, validation_features)
        calibrator = MulticlassProbabilityCalibrator(
            CalibrationConfig(method=run_config.calibration_method)
        )
        calibrator.fit(validation_raw, validation_target)
        elastic_probabilities = calibrator.transform(_probability_frame(elastic_net, test_features))

        plain_logistic = PlainMultinomialLogisticModel(plain_logistic_config)
        plain_logistic.fit(training_features, training_target)
        plain_probabilities = _probability_frame(plain_logistic, test_features)

        unconditional_probabilities = _unconditional_probabilities(training_target, test_index)
        recency_vector = recency_weighted_class_probabilities(
            training_target,
            config=RecencyWeightedBaselineConfig(half_life=run_config.recency_half_life),
        )
        recency_probabilities = constant_probability_frame(recency_vector, test_index)

        benchmark_probabilities = {
            "unconditional": unconditional_probabilities,
            "recency_weighted": recency_probabilities,
            "plain_logistic": plain_probabilities,
            "elastic_net": elastic_probabilities,
        }
        benchmark_metrics = BenchmarkMetrics(
            unconditional=_evaluate_metrics(
                test_target, unconditional_probabilities, test_metadata, economic_evaluator
            ),
            recency_weighted=_evaluate_metrics(
                test_target, recency_probabilities, test_metadata, economic_evaluator
            ),
            plain_logistic=_evaluate_metrics(
                test_target, plain_probabilities, test_metadata, economic_evaluator
            ),
            elastic_net=_evaluate_metrics(
                test_target, elastic_probabilities, test_metadata, economic_evaluator
            ),
        )
        approval = evaluate_statistical_approval(
            benchmark_metrics.elastic_net,
            benchmark_metrics.unconditional,
            criteria,
        )
        fold_results.append(
            FoldMetrics(
                fold_number=fold_number,
                model=benchmark_metrics.elastic_net,
                baseline=benchmark_metrics.unconditional,
                statistical_approval=approval,
                benchmarks=benchmark_metrics,
            )
        )
        aggregate_targets.append(test_target)
        aggregate_metadata.append(test_metadata)
        for name, probabilities in benchmark_probabilities.items():
            aggregate_probabilities[name].append(probabilities)

    combined_target = pd.concat(aggregate_targets)
    combined_metadata = pd.concat(aggregate_metadata)
    combined_metrics = {
        name: _evaluate_metrics(
            combined_target,
            pd.concat(probability_frames),
            combined_metadata,
            economic_evaluator,
        )
        for name, probability_frames in aggregate_probabilities.items()
    }
    aggregate_benchmarks = BenchmarkMetrics(
        unconditional=combined_metrics["unconditional"],
        recency_weighted=combined_metrics["recency_weighted"],
        plain_logistic=combined_metrics["plain_logistic"],
        elastic_net=combined_metrics["elastic_net"],
    )
    aggregate_model = aggregate_benchmarks.elastic_net
    aggregate_baseline = aggregate_benchmarks.unconditional
    aggregate_approval = evaluate_statistical_approval(
        aggregate_model,
        aggregate_baseline,
        criteria,
    )
    all_folds_approved = all(item.statistical_approval.approved for item in fold_results)
    statistically_approved = aggregate_approval.approved and all_folds_approved
    reasons = list(aggregate_approval.reasons)
    if not all_folds_approved:
        reasons.append("one_or_more_walk_forward_folds_failed")
    if aggregate_benchmarks.best_brier_name() != "elastic_net":
        statistically_approved = False
        reasons.append("elastic_net_not_best_brier_benchmark")

    if economic_evaluator is None:
        approved = False
        reasons.append("economic_evaluator_not_configured")
    else:
        minimum_economic_value = (
            criteria.minimum_economic_value if isinstance(criteria, ApprovalCriteria) else 0.0
        )
        economic_approved = aggregate_model.economic_value > minimum_economic_value
        approved = statistically_approved and economic_approved
        if not economic_approved:
            reasons.append("economic_value_not_positive")

    return EmpiricalRunResult(
        folds=tuple(fold_results),
        statistically_approved=statistically_approved,
        approved=approved,
        approval_reasons=tuple(dict.fromkeys(reasons)),
        aggregate_model=aggregate_model,
        aggregate_baseline=aggregate_baseline,
        aggregate_benchmarks=aggregate_benchmarks,
    )
