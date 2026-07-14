"""Purged training-only selection for Elastic-Net configuration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ung_forecast.models.elastic_net import ElasticNetConfig, ElasticNetMultinomialModel
from ung_forecast.schemas import ProbabilityForecast
from ung_forecast.validation.metrics import CLASS_ORDER, multiclass_brier_score
from ung_forecast.validation.purge import purge_overlapping_training_rows


@dataclass(frozen=True, slots=True)
class ElasticNetSelectionConfig:
    candidates: tuple[ElasticNetConfig, ...] = (
        ElasticNetConfig(c=0.1, l1_ratio=0.0),
        ElasticNetConfig(c=0.1, l1_ratio=0.5),
        ElasticNetConfig(c=0.1, l1_ratio=1.0),
        ElasticNetConfig(c=1.0, l1_ratio=0.0),
        ElasticNetConfig(c=1.0, l1_ratio=0.5),
        ElasticNetConfig(c=1.0, l1_ratio=1.0),
        ElasticNetConfig(c=10.0, l1_ratio=0.0),
        ElasticNetConfig(c=10.0, l1_ratio=0.5),
        ElasticNetConfig(c=10.0, l1_ratio=1.0),
    )
    validation_fraction: float = 0.2
    minimum_training_rows: int = 300
    minimum_validation_rows: int = 100

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("Elastic-Net candidate grid cannot be empty")
        keys = {(item.c, item.l1_ratio, item.max_iter, item.random_state) for item in self.candidates}
        if len(keys) != len(self.candidates):
            raise ValueError("Elastic-Net candidate grid contains duplicates")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between zero and one")
        if self.minimum_training_rows <= 0 or self.minimum_validation_rows <= 0:
            raise ValueError("Elastic-Net selection row minimums must be positive")


@dataclass(frozen=True, slots=True)
class ElasticNetCandidateScore:
    config: ElasticNetConfig
    brier_score: float
    training_rows: int
    validation_rows: int


@dataclass(frozen=True, slots=True)
class ElasticNetSelectionResult:
    selected: ElasticNetConfig
    candidates: tuple[ElasticNetCandidateScore, ...]
    training_index: pd.DatetimeIndex
    validation_index: pd.DatetimeIndex


def _probability_frame(
    forecasts: list[ProbabilityForecast],
    index: pd.Index,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "LOWER_FIRST": item.lower_first,
                "UPPER_FIRST": item.upper_first,
                "NEITHER": item.neither,
            }
            for item in forecasts
        ],
        index=index,
        columns=CLASS_ORDER,
    )


def select_elastic_net_config(
    features: pd.DataFrame,
    target: pd.Series,
    label_end_time: pd.Series,
    outer_training_index: pd.DatetimeIndex,
    *,
    config: ElasticNetSelectionConfig,
) -> ElasticNetSelectionResult:
    """Select configuration using only a purged chronological split of outer training rows."""

    if outer_training_index.empty:
        raise ValueError("Outer training index cannot be empty")
    proposed_validation_rows = int(len(outer_training_index) * config.validation_fraction)
    validation_rows = max(config.minimum_validation_rows, proposed_validation_rows)
    if len(outer_training_index) - validation_rows < config.minimum_training_rows:
        raise ValueError("Outer training interval is too short for Elastic-Net selection")

    validation_index = pd.DatetimeIndex(outer_training_index[-validation_rows:])
    candidate_training_index = pd.DatetimeIndex(outer_training_index[:-validation_rows])
    training_index = purge_overlapping_training_rows(
        candidate_training_index,
        label_end_time,
        evaluation_start=validation_index[0],
    )
    if len(training_index) < config.minimum_training_rows:
        raise ValueError("Purged Elastic-Net selection training block is too short")

    training_target = target.loc[training_index]
    validation_target = target.loc[validation_index]
    required_classes = set(CLASS_ORDER)
    if set(training_target.astype(str).unique()) != required_classes:
        raise ValueError("Elastic-Net selection training block lacks required classes")
    if set(validation_target.astype(str).unique()) != required_classes:
        raise ValueError("Elastic-Net selection validation block lacks required classes")

    scores: list[ElasticNetCandidateScore] = []
    for candidate in config.candidates:
        model = ElasticNetMultinomialModel(candidate)
        model.fit(features.loc[training_index], training_target)
        probabilities = _probability_frame(
            model.predict_probabilities(features.loc[validation_index]), validation_index
        )
        scores.append(
            ElasticNetCandidateScore(
                config=candidate,
                brier_score=multiclass_brier_score(validation_target, probabilities),
                training_rows=len(training_index),
                validation_rows=len(validation_index),
            )
        )

    selected_score = min(
        scores,
        key=lambda item: (item.brier_score, item.config.c, item.config.l1_ratio),
    )
    return ElasticNetSelectionResult(
        selected=selected_score.config,
        candidates=tuple(scores),
        training_index=training_index,
        validation_index=validation_index,
    )
