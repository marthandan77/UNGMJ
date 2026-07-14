"""Multinomial elastic-net logistic-regression baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ung_forecast.schemas import OutcomeClass, ProbabilityForecast


@dataclass(frozen=True, slots=True)
class ElasticNetConfig:
    c: float = 1.0
    l1_ratio: float = 0.5
    max_iter: int = 5000
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.c <= 0:
            raise ValueError("c must be positive")
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be between zero and one")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")


class ElasticNetMultinomialModel:
    def __init__(self, config: ElasticNetConfig | None = None) -> None:
        self.config = config or ElasticNetConfig()
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        C=self.config.c,
                        l1_ratio=self.config.l1_ratio,
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )
        self.feature_names: tuple[str, ...] | None = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        if features.empty:
            raise ValueError("features cannot be empty")
        if len(features) != len(target):
            raise ValueError("features and target lengths must match")
        if features.isna().any().any():
            raise ValueError("features contain missing values")
        observed = set(target.astype(str).unique())
        expected = {outcome.value for outcome in OutcomeClass}
        if observed != expected:
            raise ValueError(f"Training target must contain all outcomes: {sorted(expected)}")

        self.feature_names = tuple(features.columns)
        self.pipeline.fit(features, target.astype(str))

    def predict_probabilities(self, features: pd.DataFrame) -> list[ProbabilityForecast]:
        if self.feature_names is None:
            raise RuntimeError("Model has not been fitted")
        if tuple(features.columns) != self.feature_names:
            raise ValueError("Feature schema does not match fitted model")
        if features.isna().any().any():
            raise ValueError("features contain missing values")

        raw = self.pipeline.predict_proba(features)
        classifier = self.pipeline.named_steps["model"]
        classes = [str(value) for value in classifier.classes_]
        index_by_class = {label: index for index, label in enumerate(classes)}

        forecasts: list[ProbabilityForecast] = []
        for row in np.asarray(raw, dtype=float):
            forecasts.append(
                ProbabilityForecast(
                    lower_first=float(row[index_by_class[OutcomeClass.LOWER_FIRST.value]]),
                    upper_first=float(row[index_by_class[OutcomeClass.UPPER_FIRST.value]]),
                    neither=float(row[index_by_class[OutcomeClass.NEITHER.value]]),
                )
            )
        return forecasts

    def coefficient_frame(self) -> pd.DataFrame:
        if self.feature_names is None:
            raise RuntimeError("Model has not been fitted")
        classifier = self.pipeline.named_steps["model"]
        return pd.DataFrame(
            classifier.coef_,
            index=[str(value) for value in classifier.classes_],
            columns=self.feature_names,
        )
