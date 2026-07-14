"""Predeclared probability baselines for empirical model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ung_forecast.validation.metrics import CLASS_ORDER


@dataclass(frozen=True, slots=True)
class RecencyWeightedBaselineConfig:
    """Exponentially weight recent observations without using future data."""

    half_life: float = 250.0
    minimum_probability: float = 1e-6

    def __post_init__(self) -> None:
        if self.half_life <= 0:
            raise ValueError("half_life must be positive")
        if not 0.0 < self.minimum_probability < 1.0 / len(CLASS_ORDER):
            raise ValueError("minimum_probability is outside the admissible range")


def recency_weighted_class_probabilities(
    target: pd.Series,
    *,
    config: RecencyWeightedBaselineConfig | None = None,
) -> pd.Series:
    """Estimate class probabilities using exponentially decaying sample weights.

    The newest training observation receives weight one. An observation exactly
    ``half_life`` rows older receives weight one half. No timestamps after the
    supplied training target are inspected.
    """

    effective_config = config or RecencyWeightedBaselineConfig()
    if target.empty:
        raise ValueError("target cannot be empty")

    labels = target.astype(str)
    invalid = set(labels.unique()).difference(CLASS_ORDER)
    if invalid:
        raise ValueError(f"Unknown target classes: {sorted(invalid)}")

    age = np.arange(len(labels) - 1, -1, -1, dtype=float)
    weights = np.exp2(-age / effective_config.half_life)
    weighted_counts = pd.Series(0.0, index=CLASS_ORDER, dtype=float)
    for class_name in CLASS_ORDER:
        weighted_counts.loc[class_name] = float(weights[labels.to_numpy() == class_name].sum())

    probabilities = weighted_counts / float(weighted_counts.sum())
    probabilities = probabilities.clip(lower=effective_config.minimum_probability)
    probabilities = probabilities / float(probabilities.sum())
    return probabilities.reindex(CLASS_ORDER)


def constant_probability_frame(
    probabilities: pd.Series,
    index: pd.Index,
) -> pd.DataFrame:
    """Repeat one class-probability vector over a requested evaluation index."""

    ordered = probabilities.reindex(CLASS_ORDER)
    if ordered.isna().any():
        raise ValueError("Probability vector is missing one or more classes")
    values = ordered.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probability vector must be finite and non-negative")
    if not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError("Probability vector must sum to one")
    return pd.DataFrame(
        np.tile(values, (len(index), 1)),
        index=index,
        columns=CLASS_ORDER,
    )
