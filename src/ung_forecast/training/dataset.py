"""Build leakage-safe supervised datasets from features and future paths."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ung_forecast.barriers import create_barriers
from ung_forecast.labels import LabelStatus, label_future_path
from ung_forecast.schemas import OutcomeClass


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    features: pd.DataFrame
    target: pd.Series
    label_end_time: pd.Series
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        if not self.features.index.equals(self.target.index):
            raise ValueError("Feature and target indices must match")
        if not self.features.index.equals(self.label_end_time.index):
            raise ValueError("Feature and label-end indices must match")


def build_training_dataset(
    market_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    volatility: pd.Series,
    *,
    required_bars: int,
    lower_multiplier: float,
    upper_multiplier: float,
    horizon_key: str,
) -> TrainingDataset:
    """Create labeled rows without using future information in features."""

    common = market_data.index.intersection(feature_frame.index).intersection(volatility.index)
    rows: list[pd.Timestamp] = []
    targets: list[OutcomeClass] = []
    end_times: list[pd.Timestamp] = []
    metadata_rows: list[dict[str, object]] = []

    for timestamp in common:
        feature_row = feature_frame.loc[timestamp]
        sigma = volatility.loc[timestamp]
        if feature_row.isna().any() or pd.isna(sigma) or float(sigma) <= 0:
            continue
        market_position = market_data.index.get_loc(timestamp)
        if not isinstance(market_position, int):
            raise ValueError("Market data timestamps must be unique")
        future = market_data.iloc[market_position + 1 : market_position + 1 + required_bars]
        barriers = create_barriers(
            current_price=float(market_data.loc[timestamp, "Close"]),
            volatility_estimate=float(sigma),
            lower_multiplier=lower_multiplier,
            upper_multiplier=upper_multiplier,
            horizon_key=horizon_key,
        )
        label = label_future_path(future, barriers=barriers, required_bars=required_bars)
        if label.status is not LabelStatus.VALID or label.outcome is None:
            continue
        label_end = pd.Timestamp(future.index[-1])
        rows.append(pd.Timestamp(timestamp))
        targets.append(label.outcome)
        end_times.append(label_end)
        metadata_rows.append(
            {
                "lower_price": barriers.lower_price,
                "upper_price": barriers.upper_price,
                "maximum_upward_excursion": label.maximum_upward_excursion,
                "maximum_downward_excursion": label.maximum_downward_excursion,
                "time_to_event_bars": label.time_to_event_bars,
            }
        )

    index = pd.DatetimeIndex(rows)
    return TrainingDataset(
        features=feature_frame.loc[index].copy(),
        target=pd.Series([value.value for value in targets], index=index, name="target"),
        label_end_time=pd.Series(end_times, index=index, name="label_end_time"),
        metadata=pd.DataFrame(metadata_rows, index=index),
    )
