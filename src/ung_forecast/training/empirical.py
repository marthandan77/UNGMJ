"""Empirical walk-forward evaluation for one prebuilt horizon dataset.

The runner never fabricates economic approval. If no realized economic-value
evaluator is supplied, the horizon remains research-only even when its
probability metrics improve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ung_forecast.models.calibration import CalibrationConfig, MulticlassProbabilityCalibrator
from ung_forecast.models.elastic_net import ElasticNetConfig, ElasticNetMultinomialModel
from ung_forecast.training.dataset import TrainingDataset
from ung_forecast.validation.approval import (
    ApprovalCriteria,
    ApprovalDecision,
    ValidationMetrics,
    evaluate_approval,
)
from ung_forecast.validation.metrics import (
    CLASS_ORDER,
    class_frequency_baseline,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
)
from ung_forecast.validation.purge import purge_overlapping_training_rows
from ung_forecast.validation.walk_forward import WalkForwardFold, generate_walk_forward_folds

EconomicValueEvaluator = Callable[[pd.Series, pd.DataFrame, pd.DataFrame], float]


@dataclass(frozen=True, slots=True)
class EmpiricalRunConfig:
    minimum_train_size: int
    validation_size: int
    test_size: int
    purge_size: int
    embargo_size: int
    calibration_method: str = "platt"


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    fold_number: int
    model: ValidationMetrics
    baseline: ValidationMetrics
    approval: ApprovalDecision


@dataclass(frozen=True, slots=True)
class EmpiricalRunResult:
    folds: tuple[FoldMetrics, ...]
    approved: bool
    approval_reasons: tuple[str, ...]
    aggregate_model: ValidationMetrics
    aggregate_baseline: ValidationMetrics


def _probability_frame(model: ElasticNetMultinomialModel, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for probability in model.predict_probabilities(features):
        rows.append(
            {
                "LOWER_FIRST": probability.lower_first,
                "UPPER_FIRST": probability.upper_first,
                "NEITHER": probability.neither,
            }
        )
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


def _safe_approval(
    model_metrics: ValidationMetrics,
    baseline_metrics: ValidationMetrics,
    criteria: ApprovalCriteria,
    economic_evaluator: EconomicValueEvaluator | None,
) -> ApprovalDecision:
    if economic_evaluator is None:
        return ApprovalDecision(False, ("economic_evaluator_not_configured",))
    return evaluate_approval(model_metrics, baseline_metrics, criteria)


def run_empirical_evaluation(
    dataset: TrainingDataset,
    *,
    run_config: EmpiricalRunConfig,
    model_config: ElasticNetConfig | None = None,
    approval_criteria: ApprovalCriteria | None = None,
    economic_evaluator: EconomicValueEvaluator | None = None,
) -> EmpiricalRunResult:
    criteria = approval_criteria or ApprovalCriteria()
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
    aggregate_model_probabilities: list[pd.DataFrame] = []
    aggregate_baseline_probabilities: list[pd.DataFrame] = []
    aggregate_metadata: list[pd.DataFrame] = []

    for fold_number, fold in enumerate(folds, start=1):
        train_index = dataset.features.index[list(fold.train)]
        validation_index = dataset.features.index[list(fold.validation)]
        test_index = dataset.features.index[list(fold.test)]
        train_index = purge_overlapping_training_rows(
            train_index,
            dataset.label_end_time,
            evaluation_start=validation_index[0],
        )
        if train_index.empty:
            raise ValueError("Purging removed every training observation")

        model = ElasticNetMultinomialModel(model_config)
        model.fit(dataset.features.loc[train_index], dataset.target.loc[train_index])

        validation_raw = _probability_frame(model, dataset.features.loc[validation_index])
        calibrator = MulticlassProbabilityCalibrator(
            CalibrationConfig(method=run_config.calibration_method)
        )
        calibrator.fit(validation_raw, dataset.target.loc[validation_index])

        test_raw = _probability_frame(model, dataset.features.loc[test_index])
        test_probabilities = calibrator.transform(test_raw)
        test_target = dataset.target.loc[test_index]
        test_metadata = dataset.metadata.loc[test_index]
        baseline_probabilities = class_frequency_baseline(dataset.target.loc[train_index]).iloc[
            :1
        ]
        baseline_probabilities = pd.DataFrame(
            np.tile(baseline_probabilities.iloc[0].to_numpy(), (len(test_index), 1)),
            index=test_index,
            columns=CLASS_ORDER,
        )

        model_metrics = _evaluate_metrics(
            test_target, test_probabilities, test_metadata, economic_evaluator
        )
        baseline_metrics = _evaluate_metrics(
            test_target, baseline_probabilities, test_metadata, economic_evaluator
        )
        approval = _safe_approval(
            model_metrics, baseline_metrics, criteria, economic_evaluator
        )
        fold_results.append(FoldMetrics(fold_number, model_metrics, baseline_metrics, approval))
        aggregate_targets.append(test_target)
        aggregate_model_probabilities.append(test_probabilities)
        aggregate_baseline_probabilities.append(baseline_probabilities)
        aggregate_metadata.append(test_metadata)

    combined_target = pd.concat(aggregate_targets)
    combined_model = pd.concat(aggregate_model_probabilities)
    combined_baseline = pd.concat(aggregate_baseline_probabilities)
    combined_metadata = pd.concat(aggregate_metadata)
    aggregate_model_metrics = _evaluate_metrics(
        combined_target, combined_model, combined_metadata, economic_evaluator
    )
    aggregate_baseline_metrics = _evaluate_metrics(
        combined_target, combined_baseline, combined_metadata, economic_evaluator
    )
    aggregate_approval = _safe_approval(
        aggregate_model_metrics,
        aggregate_baseline_metrics,
        criteria,
        economic_evaluator,
    )
    all_folds_approved = all(result.approval.approved for result in fold_results)
    approved = aggregate_approval.approved and all_folds_approved
    reasons = list(aggregate_approval.reasons)
    if not all_folds_approved:
        reasons.append("one_or_more_walk_forward_folds_failed")

    return EmpiricalRunResult(
        folds=tuple(fold_results),
        approved=approved,
        approval_reasons=tuple(dict.fromkeys(reasons)),
        aggregate_model=aggregate_model_metrics,
        aggregate_baseline=aggregate_baseline_metrics,
    )
