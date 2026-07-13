"""Fold-only selection of volatility-scaled barrier multipliers.

Candidate barriers are ranked using training and validation observations only.
The untouched test interval is never accepted by this API.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ung_forecast.training.dataset import TrainingDataset, build_training_dataset
from ung_forecast.validation.baselines import constant_probability_frame
from ung_forecast.validation.metrics import CLASS_ORDER, multiclass_brier_score
from ung_forecast.validation.purge import purge_overlapping_training_rows


@dataclass(frozen=True, slots=True)
class BarrierCandidate:
    lower_multiplier: float
    upper_multiplier: float

    def __post_init__(self) -> None:
        if self.lower_multiplier <= 0 or self.upper_multiplier <= 0:
            raise ValueError("Barrier multipliers must be positive")


@dataclass(frozen=True, slots=True)
class BarrierSelectionConfig:
    candidates: tuple[BarrierCandidate, ...]
    minimum_training_rows: int = 100
    minimum_validation_rows: int = 30
    require_all_classes: bool = True

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("At least one barrier candidate is required")
        if self.minimum_training_rows <= 0 or self.minimum_validation_rows <= 0:
            raise ValueError("Minimum row requirements must be positive")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Barrier candidates must be unique")


@dataclass(frozen=True, slots=True)
class BarrierCandidateResult:
    candidate: BarrierCandidate
    accepted: bool
    validation_brier: float | None
    training_rows: int
    validation_rows: int
    training_classes: tuple[str, ...]
    validation_classes: tuple[str, ...]
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class BarrierSelectionResult:
    selected: BarrierCandidate
    candidates: tuple[BarrierCandidateResult, ...]


def _class_tuple(target: pd.Series) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in target.astype(str).unique()))


def _candidate_dataset(
    market_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    volatility: pd.Series,
    *,
    candidate: BarrierCandidate,
    required_bars: int,
    horizon_key: str,
) -> TrainingDataset:
    return build_training_dataset(
        market_data,
        feature_frame,
        volatility,
        required_bars=required_bars,
        lower_multiplier=candidate.lower_multiplier,
        upper_multiplier=candidate.upper_multiplier,
        horizon_key=horizon_key,
    )


def select_barriers_fold_only(
    market_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    volatility: pd.Series,
    *,
    training_index: pd.DatetimeIndex,
    validation_index: pd.DatetimeIndex,
    validation_end: pd.Timestamp,
    required_bars: int,
    horizon_key: str,
    config: BarrierSelectionConfig,
) -> BarrierSelectionResult:
    """Select barriers without accepting or inspecting a test interval.

    Ranking is lexicographic and deterministic:
    1. lowest validation Brier score from a training-frequency forecast;
    2. largest number of usable validation labels;
    3. smallest lower multiplier;
    4. smallest upper multiplier.

    Training rows whose labels reach validation are purged. Validation rows whose
    label end exceeds ``validation_end`` are excluded, so no future path can cross
    either evaluation boundary.
    """

    if training_index.empty or validation_index.empty:
        raise ValueError("Training and validation indices cannot be empty")
    if required_bars <= 0:
        raise ValueError("required_bars must be positive")
    if training_index.max() >= validation_index.min():
        raise ValueError("Training observations must precede validation observations")

    outcomes: list[BarrierCandidateResult] = []
    ranked: list[tuple[float, int, float, float, BarrierCandidate]] = []
    expected_classes = set(CLASS_ORDER)

    for candidate in config.candidates:
        dataset = _candidate_dataset(
            market_data,
            feature_frame,
            volatility,
            candidate=candidate,
            required_bars=required_bars,
            horizon_key=horizon_key,
        )
        train_rows = pd.DatetimeIndex(
            dataset.features.index.intersection(training_index)
        )
        train_rows = purge_overlapping_training_rows(
            train_rows,
            dataset.label_end_time,
            evaluation_start=validation_index[0],
        )
        validation_rows = dataset.features.index.intersection(validation_index)
        if len(validation_rows):
            valid_end = pd.to_datetime(dataset.label_end_time.loc[validation_rows]) <= validation_end
            validation_rows = validation_rows[valid_end.to_numpy()]

        training_target = dataset.target.loc[train_rows]
        validation_target = dataset.target.loc[validation_rows]
        training_classes = _class_tuple(training_target)
        validation_classes = _class_tuple(validation_target)

        reason: str | None = None
        if len(train_rows) < config.minimum_training_rows:
            reason = "insufficient_training_rows"
        elif len(validation_rows) < config.minimum_validation_rows:
            reason = "insufficient_validation_rows"
        elif config.require_all_classes and set(training_classes) != expected_classes:
            reason = "training_missing_class"
        elif config.require_all_classes and set(validation_classes) != expected_classes:
            reason = "validation_missing_class"

        if reason is not None:
            outcomes.append(
                BarrierCandidateResult(
                    candidate=candidate,
                    accepted=False,
                    validation_brier=None,
                    training_rows=len(train_rows),
                    validation_rows=len(validation_rows),
                    training_classes=training_classes,
                    validation_classes=validation_classes,
                    rejection_reason=reason,
                )
            )
            continue

        frequencies = (
            training_target.astype(str)
            .value_counts(normalize=True)
            .reindex(CLASS_ORDER, fill_value=0.0)
        )
        validation_probabilities = constant_probability_frame(frequencies, validation_rows)
        brier = multiclass_brier_score(validation_target, validation_probabilities)
        outcomes.append(
            BarrierCandidateResult(
                candidate=candidate,
                accepted=True,
                validation_brier=brier,
                training_rows=len(train_rows),
                validation_rows=len(validation_rows),
                training_classes=training_classes,
                validation_classes=validation_classes,
                rejection_reason=None,
            )
        )
        ranked.append(
            (
                brier,
                -len(validation_rows),
                candidate.lower_multiplier,
                candidate.upper_multiplier,
                candidate,
            )
        )

    if not ranked:
        raise ValueError("No barrier candidate passed fold-only selection requirements")
    ranked.sort(key=lambda item: item[:-1])
    return BarrierSelectionResult(selected=ranked[0][-1], candidates=tuple(outcomes))
