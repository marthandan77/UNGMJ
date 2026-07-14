"""Probabilistic validation metrics for three-class forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ung_forecast.schemas import OutcomeClass

CLASS_ORDER = [outcome.value for outcome in OutcomeClass]


def probability_frame(rows: list[dict[str, float]], index: pd.Index | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(rows, index=index)
    missing = set(CLASS_ORDER).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing probability columns: {sorted(missing)}")
    frame = frame.loc[:, CLASS_ORDER].astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Probabilities must be finite")
    if (frame < 0).any().any() or (frame > 1).any().any():
        raise ValueError("Probabilities must be within [0, 1]")
    if not np.allclose(frame.sum(axis=1).to_numpy(), 1.0, atol=1e-8):
        raise ValueError("Each probability row must sum to one")
    return frame


def multiclass_brier_score(target: pd.Series, probabilities: pd.DataFrame) -> float:
    aligned = probabilities.reindex(target.index)
    if aligned.isna().any().any():
        raise ValueError("Target and probability indices must align")
    one_hot = pd.get_dummies(target.astype(str)).reindex(columns=CLASS_ORDER, fill_value=0)
    return float(np.mean(np.sum((aligned.to_numpy() - one_hot.to_numpy()) ** 2, axis=1)))


def multiclass_log_loss(target: pd.Series, probabilities: pd.DataFrame) -> float:
    aligned = probabilities.reindex(target.index).loc[:, CLASS_ORDER].astype(float)
    if aligned.isna().any().any():
        raise ValueError("Target and probability indices must align")
    target_values = target.astype(str)
    invalid = set(target_values.unique()).difference(CLASS_ORDER)
    if invalid:
        raise ValueError(f"Unknown target classes: {sorted(invalid)}")
    class_positions = {class_name: position for position, class_name in enumerate(CLASS_ORDER)}
    target_positions = np.fromiter(
        (class_positions[class_name] for class_name in target_values.to_numpy()),
        dtype=np.int64,
        count=len(target_values),
    )
    matrix = aligned.to_numpy(dtype=float)
    selected = matrix[np.arange(len(matrix)), target_positions]
    selected = np.clip(selected, 1e-15, 1.0)
    return float(-np.log(selected).mean())


def expected_calibration_error(
    target: pd.Series,
    probabilities: pd.DataFrame,
    *,
    bins: int = 10,
) -> float:
    if bins <= 1:
        raise ValueError("bins must exceed one")
    aligned = probabilities.reindex(target.index)
    predicted_class = aligned.idxmax(axis=1).astype(str)
    confidence = aligned.max(axis=1).astype(float)
    target_values = target.astype(str)
    correctness = pd.Series(
        predicted_class.to_numpy() == target_values.to_numpy(),
        index=target.index,
        dtype=float,
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    total = len(target)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        error += count / total * abs(float(correctness[mask].mean() - confidence[mask].mean()))
    return float(error)


def class_frequency_baseline(target: pd.Series) -> pd.DataFrame:
    frequencies = target.astype(str).value_counts(normalize=True).reindex(CLASS_ORDER, fill_value=0.0)
    data = np.tile(frequencies.to_numpy(), (len(target), 1))
    return pd.DataFrame(data, index=target.index, columns=CLASS_ORDER)
