"""Deterministic diagnostics for model and feature stability across folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureDriftDiagnostics:
    validation_mahalanobis: float
    test_mahalanobis: float


@dataclass(frozen=True, slots=True)
class CoefficientDiagnostics:
    l2_norm: float
    nonzero_count: int
    total_count: int


def _mean_mahalanobis_distance(reference: pd.DataFrame, comparison: pd.DataFrame) -> float:
    if reference.empty or comparison.empty:
        raise ValueError("Feature drift diagnostics require non-empty frames")
    if tuple(reference.columns) != tuple(comparison.columns):
        raise ValueError("Feature drift frames must use the same ordered columns")
    reference_values = reference.astype(float).to_numpy()
    comparison_values = comparison.astype(float).to_numpy()
    if not np.isfinite(reference_values).all() or not np.isfinite(comparison_values).all():
        raise ValueError("Feature drift frames must contain only finite values")

    mean_difference = comparison_values.mean(axis=0) - reference_values.mean(axis=0)
    covariance = np.cov(reference_values, rowvar=False, ddof=0)
    covariance_matrix = np.atleast_2d(np.asarray(covariance, dtype=float))
    inverse = np.linalg.pinv(covariance_matrix, hermitian=True)
    squared_distance = float(mean_difference @ inverse @ mean_difference)
    return float(np.sqrt(max(squared_distance, 0.0)))


def feature_drift_diagnostics(
    training_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> FeatureDriftDiagnostics:
    return FeatureDriftDiagnostics(
        validation_mahalanobis=_mean_mahalanobis_distance(
            training_features, validation_features
        ),
        test_mahalanobis=_mean_mahalanobis_distance(training_features, test_features),
    )


def coefficient_diagnostics(coefficients: pd.DataFrame) -> CoefficientDiagnostics:
    if coefficients.empty:
        raise ValueError("Coefficient diagnostics require a non-empty frame")
    values = coefficients.astype(float).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Coefficient frame must contain only finite values")
    return CoefficientDiagnostics(
        l2_norm=float(np.linalg.norm(values.ravel(), ord=2)),
        nonzero_count=int(np.count_nonzero(values)),
        total_count=int(values.size),
    )
