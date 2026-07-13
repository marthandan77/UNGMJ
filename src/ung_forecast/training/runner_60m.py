"""Deterministic orchestration for the 60-minute research horizon.

This module plans leakage-safe folds and selects volatility-scaled barriers using
training and validation observations only. It deliberately does not train or
approve a model; evaluation is layered on after this orchestration is verified.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ung_forecast.training.barrier_selection import (
    BarrierCandidate,
    BarrierSelectionConfig,
    BarrierSelectionResult,
    select_barriers_fold_only,
)
from ung_forecast.training.dataset import TrainingDataset, build_training_dataset
from ung_forecast.validation.purge import purge_overlapping_training_rows
from ung_forecast.validation.walk_forward import WalkForwardFold, generate_walk_forward_folds

REQUIRED_5M_BARS = 12
HORIZON_KEY = "60m"


@dataclass(frozen=True, slots=True)
class SixtyMinuteRunnerConfig:
    minimum_train_size: int
    validation_size: int
    test_size: int
    purge_size: int
    embargo_size: int
    barrier_selection: BarrierSelectionConfig


@dataclass(frozen=True, slots=True)
class SixtyMinuteFoldPlan:
    fold_number: int
    selected_barrier: BarrierCandidate
    selection: BarrierSelectionResult
    dataset: TrainingDataset
    training_index: pd.DatetimeIndex
    validation_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


@dataclass(frozen=True, slots=True)
class SixtyMinuteRunPlan:
    folds: tuple[SixtyMinuteFoldPlan, ...]


def _eligible_feature_index(
    market_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    volatility: pd.Series,
) -> pd.DatetimeIndex:
    if not isinstance(market_data.index, pd.DatetimeIndex):
        raise ValueError("market_data requires a DatetimeIndex")
    if not isinstance(feature_frame.index, pd.DatetimeIndex):
        raise ValueError("feature_frame requires a DatetimeIndex")
    if not isinstance(volatility.index, pd.DatetimeIndex):
        raise ValueError("volatility requires a DatetimeIndex")
    if market_data.index.has_duplicates or not market_data.index.is_monotonic_increasing:
        raise ValueError("market_data index must be unique and increasing")

    common = market_data.index.intersection(feature_frame.index).intersection(volatility.index)
    if len(common) <= REQUIRED_5M_BARS:
        raise ValueError("Insufficient aligned rows for the 60-minute horizon")

    last_eligible_position = len(market_data) - REQUIRED_5M_BARS - 1
    eligible_market_index = market_data.index[: last_eligible_position + 1]
    eligible = pd.DatetimeIndex(common.intersection(eligible_market_index))
    valid_features = ~feature_frame.loc[eligible].isna().any(axis=1)
    valid_volatility = volatility.loc[eligible].notna() & (volatility.loc[eligible] > 0)
    return pd.DatetimeIndex(eligible[valid_features.to_numpy() & valid_volatility.to_numpy()])


def _fold_indices(index: pd.DatetimeIndex, fold: WalkForwardFold) -> tuple[pd.DatetimeIndex, ...]:
    return (
        pd.DatetimeIndex(index[list(fold.train)]),
        pd.DatetimeIndex(index[list(fold.validation)]),
        pd.DatetimeIndex(index[list(fold.test)]),
    )


def build_sixty_minute_run_plan(
    market_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    volatility: pd.Series,
    *,
    config: SixtyMinuteRunnerConfig,
) -> SixtyMinuteRunPlan:
    """Build fold plans without exposing test observations to barrier selection."""

    eligible_index = _eligible_feature_index(market_data, feature_frame, volatility)
    folds = generate_walk_forward_folds(
        sample_count=len(eligible_index),
        minimum_train_size=config.minimum_train_size,
        validation_size=config.validation_size,
        test_size=config.test_size,
        purge_size=config.purge_size,
        embargo_size=config.embargo_size,
    )

    plans: list[SixtyMinuteFoldPlan] = []
    for fold_number, fold in enumerate(folds, start=1):
        training_index, validation_index, test_index = _fold_indices(eligible_index, fold)
        selection = select_barriers_fold_only(
            market_data,
            feature_frame,
            volatility,
            training_index=training_index,
            validation_index=validation_index,
            validation_end=validation_index[-1],
            required_bars=REQUIRED_5M_BARS,
            horizon_key=HORIZON_KEY,
            config=config.barrier_selection,
        )
        selected = selection.selected
        dataset = build_training_dataset(
            market_data,
            feature_frame,
            volatility,
            required_bars=REQUIRED_5M_BARS,
            lower_multiplier=selected.lower_multiplier,
            upper_multiplier=selected.upper_multiplier,
            horizon_key=HORIZON_KEY,
        )

        selected_training = pd.DatetimeIndex(dataset.features.index.intersection(training_index))
        selected_training = purge_overlapping_training_rows(
            selected_training,
            dataset.label_end_time,
            evaluation_start=validation_index[0],
        )
        selected_validation = pd.DatetimeIndex(dataset.features.index.intersection(validation_index))
        selected_validation = pd.DatetimeIndex(
            selected_validation[
                (
                    pd.to_datetime(dataset.label_end_time.loc[selected_validation])
                    <= validation_index[-1]
                ).to_numpy()
            ]
        )
        selected_test = pd.DatetimeIndex(dataset.features.index.intersection(test_index))

        if selected_training.empty or selected_validation.empty or selected_test.empty:
            raise ValueError("Selected barriers produced an unusable walk-forward fold")

        plans.append(
            SixtyMinuteFoldPlan(
                fold_number=fold_number,
                selected_barrier=selected,
                selection=selection,
                dataset=dataset,
                training_index=selected_training,
                validation_index=selected_validation,
                test_index=selected_test,
            )
        )

    if not plans:
        raise ValueError("No 60-minute walk-forward folds were generated")
    return SixtyMinuteRunPlan(folds=tuple(plans))
