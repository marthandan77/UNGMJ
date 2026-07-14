from __future__ import annotations

import numpy as np
import pandas as pd

from ung_forecast.features.engine import build_feature_frame
from ung_forecast.horizons import HorizonKey


def market_frame(days: int = 30, bars_per_day: int = 12) -> pd.DataFrame:
    timestamps: list[pd.Timestamp] = []
    for day in pd.bdate_range("2026-05-01", periods=days):
        start = pd.Timestamp(day.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        timestamps.extend(pd.date_range(start, periods=bars_per_day, freq="5min"))
    index = pd.DatetimeIndex(timestamps)
    step = np.arange(len(index), dtype=float)
    close = 10.0 + 0.002 * step + 0.05 * np.sin(step / 5.0)
    return pd.DataFrame(
        {
            "Open": close - 0.01,
            "High": close + 0.03,
            "Low": close - 0.03,
            "Close": close,
            "Volume": 1000 + (step % bars_per_day) * 20 + (step % 7) * 5,
        },
        index=index,
    )


def test_training_and_live_paths_use_same_engine() -> None:
    primary = market_frame()
    training_result = build_feature_frame(primary, horizon=HorizonKey.MINUTES_60)
    live_result = build_feature_frame(primary, horizon=HorizonKey.MINUTES_60)

    pd.testing.assert_frame_equal(training_result.frame, live_result.frame)
    pd.testing.assert_series_equal(
        training_result.max_source_timestamp,
        live_result.max_source_timestamp,
    )


def test_future_data_change_does_not_change_prior_features() -> None:
    primary = market_frame()
    original = build_feature_frame(primary, horizon=HorizonKey.MINUTES_60).frame

    changed = primary.copy()
    changed.iloc[-1, changed.columns.get_loc("Close")] += 5.0
    changed.iloc[-1, changed.columns.get_loc("High")] += 5.0
    revised = build_feature_frame(changed, horizon=HorizonKey.MINUTES_60).frame

    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])


def test_source_timestamp_audit_never_exceeds_output_timestamp() -> None:
    result = build_feature_frame(market_frame(), horizon=HorizonKey.HOURS_4)
    result.assert_no_future_sources()
    output_time = pd.Series(result.frame.index, index=result.frame.index)
    assert (result.max_source_timestamp <= output_time).all()


def test_comparison_feature_requires_exact_alignment() -> None:
    primary = market_frame()
    comparison = market_frame().iloc[1:].copy()

    try:
        build_feature_frame(
            primary,
            horizon=HorizonKey.MINUTES_60,
            comparison=comparison,
        )
    except ValueError as exc:
        assert "synchronized" in str(exc)
    else:
        raise AssertionError("Expected exact timestamp alignment failure")


def test_comparison_feature_is_included_when_aligned() -> None:
    primary = market_frame(days=40)
    comparison = primary.copy()
    comparison["Close"] = comparison["Close"] * 0.8 + 1.5
    comparison["Open"] = comparison["Close"] - 0.01
    comparison["High"] = comparison["Close"] + 0.03
    comparison["Low"] = comparison["Close"] - 0.03

    result = build_feature_frame(
        primary,
        horizon=HorizonKey.MINUTES_60,
        comparison=comparison,
    )

    assert result.includes_comparison is True
    assert "ung_ng_residual_z" in result.frame.columns
