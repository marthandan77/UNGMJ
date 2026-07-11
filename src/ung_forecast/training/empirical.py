"""Empirical walk-forward evaluation for one prebuilt horizon dataset.

This runner performs statistical approval only. Trading approval is a separate
stage requiring an explicit execution model and shadow observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ung_forecast.models.calibration import CalibrationConfig, MulticlassProbabilityCalibrator
from ung_forecast.models.elastic_net import ElasticNetConfig, ElasticNetMultinomialModel
from ung_forecast.training.dataset import TrainingDataset
from ung_forecast.validation.approval import (
    ApprovalDecision,
    StatisticalApprovalCriteria,
    ValidationMetrics,
    evaluate_statistical_approval,
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
    statistical_approval: ApprovalDecision


@dataclass(frozen=True, slots=True)
class EmpiricalRunResult:
    folds: tuple[FoldMetrics, ...]
    statistically_approved: bool
    approval_reasons: tuple[str, ...]
    aggregate_model: ValidationMetrics
    aggregate_baseline: ValidationMetrics


def _probability_frame(model: ElasticNetMultinomialModel, features: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "LOWER_FIRST": probability.lower_first,
            "UPPER_FIRST": probability.upper_first,
            "NEITHER": probability.neither,
        }
        for probability in model.predict_probabilities(features)
    ]
    return pd.DataFrame(rows, index=features.index, columns=CLASS_ORDER)


def _evaluate_metrics(target: pd.Series, probabilities: pd.DataFrame) -> ValidationMetrics:
    return ValidationMetrics(
        brier_score=multiclass_brier_score(target, probabilities),
        log_loss=multiclass_log_loss(target, probabilities),
        calibration_error=expected_calibration_error(target, probabilities),
        economic_value=float("nan"),
        sample_count=len(target),
    )


def run_empirical_evaluation(
    dataset: TrainingDataset,
    *,
    run_config: EmpiricalRunConfig,
    model_config: ElasticNetConfig | None = None,
    approval_criteria: StatisticalApprovalCriteria | None = None,
) -> EmpiricalRunResult:
    criteria = approval_criteria or StatisticalApprovalCriteria()
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

        test_probabilities = calibrator.transform(
            _probability_frame(model, dataset.features.loc[test_index])
        )
        test_target = dataset.target.loc[test_index]
        training_baseline = class_frequency_baseline(dataset.target.loc[train_index]).iloc[0]
        baseline_probabilities = pd.DataFrame(
            np.tile(training_baseline.to_numpy(), (len(test_index), 1)),
            index=test_index,
            columns=CLASS_ORDER,
        )
        model_metrics = _evaluate_metrics(test_target, test_probabilities)
        baseline_metrics = _evaluate_metrics(test_target, baseline_probabilities)
        approval = evaluate_statistical_approval(model_metrics, baseline_metrics, criteria)
        fold_results.append(FoldMetrics(fold_number, model_metrics, baseline_metrics, approval))
        aggregate_targets.append(test_target)
        aggregate_model_probabilities.append(test_probabilities)
        aggregate_baseline_probabilities.append(baseline_probabilities)

    combined_target = pd.concat(aggregate_targets)
    aggregate_model = _evaluate_metrics(combined_target, pd.concat(aggregate_model_probabilities))
    aggregate_baseline = _evaluate_metrics(
        combined_target, pd.concat(aggregate_baseline_probabilities)
    )
    aggregate_approval = evaluate_statistical_approval(
        aggregate_model, aggregate_baseline, criteria
    )
    all_folds_approved = all(item.statistical_approval.approved for item in fold_results)
    reasons = list(aggregate_approval.reasons)
    if not all_folds_approved:
        reasons.append("one_or_more_walk_forward_folds_failed")

    return EmpiricalRunResult(
        folds=tuple(fold_results),
        statistically_approved=aggregate_approval.approved and all_folds_approved,
        approval_reasons=tuple(dict.fromkeys(reasons)),
        aggregate_model=aggregate_model,
        aggregate_baseline=aggregate_baseline,
    )
