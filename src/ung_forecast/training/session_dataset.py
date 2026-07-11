"""Build session-accurate datasets for 1-, 2-, and 7-trading-day horizons.

Feature observations may be produced from intraday or daily inputs, but labels
are always generated from future completed daily session bars. This prevents
approximating a trading day with a fixed hourly-bar count.
"""

from __future__ import annotations

import pandas as pd

from ung_forecast.barriers import create_barriers
from ung_forecast.labels import LabelStatus, label_future_path
from ung_forecast.training.dataset import TrainingDataset


def build_session_training_dataset(
    daily_market_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    volatility: pd.Series,
    *,
    sessions_ahead: int,
    lower_multiplier: float,
    upper_multiplier: float,
    horizon_key: str,
) -> TrainingDataset:
    if sessions_ahead not in {1, 2, 7}:
        raise ValueError("sessions_ahead must be one of 1, 2, or 7")
    if not isinstance(daily_market_data.index, pd.DatetimeIndex):
        raise ValueError("daily_market_data requires a DatetimeIndex")
    if daily_market_data.index.has_duplicates or not daily_market_data.index.is_monotonic_increasing:
        raise ValueError("daily session timestamps must be unique and increasing")

    daily_by_date = {timestamp.date(): position for position, timestamp in enumerate(daily_market_data.index)}
    rows: list[pd.Timestamp] = []
    targets: list[str] = []
    end_times: list[pd.Timestamp] = []
    metadata_rows: list[dict[str, object]] = []

    common = feature_frame.index.intersection(volatility.index)
    for timestamp in common:
        feature_row = feature_frame.loc[timestamp]
        sigma = volatility.loc[timestamp]
        if feature_row.isna().any() or pd.isna(sigma) or float(sigma) <= 0:
            continue
        session_position = daily_by_date.get(pd.Timestamp(timestamp).date())
        if session_position is None:
            continue
        future = daily_market_data.iloc[
            session_position + 1 : session_position + 1 + sessions_ahead
        ]
        if len(future) < sessions_ahead:
            continue
        current_price = float(daily_market_data.iloc[session_position]["Close"])
        barriers = create_barriers(
            current_price=current_price,
            volatility_estimate=float(sigma),
            lower_multiplier=lower_multiplier,
            upper_multiplier=upper_multiplier,
            horizon_key=horizon_key,
        )
        label = label_future_path(
            future,
            barriers=barriers,
            required_bars=sessions_ahead,
        )
        if label.status is not LabelStatus.VALID or label.outcome is None:
            continue
        rows.append(pd.Timestamp(timestamp))
        targets.append(label.outcome.value)
        end_times.append(pd.Timestamp(future.index[-1]))
        metadata_rows.append(
            {
                "lower_price": barriers.lower_price,
                "upper_price": barriers.upper_price,
                "maximum_upward_excursion": label.maximum_upward_excursion,
                "maximum_downward_excursion": label.maximum_downward_excursion,
                "time_to_event_sessions": label.time_to_event_bars,
            }
        )

    index = pd.DatetimeIndex(rows)
    return TrainingDataset(
        features=feature_frame.loc[index].copy(),
        target=pd.Series(targets, index=index, name="target"),
        label_end_time=pd.Series(end_times, index=index, name="label_end_time"),
        metadata=pd.DataFrame(metadata_rows, index=index),
    )
