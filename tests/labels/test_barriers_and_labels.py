from __future__ import annotations

import pandas as pd
import pytest

from ung_forecast.barriers import create_barriers
from ung_forecast.labels import LabelStatus, label_future_path
from ung_forecast.schemas import OutcomeClass


def future_path(highs: list[float], lows: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-07-10 10:35", periods=len(highs), freq="5min", tz="America/New_York")
    return pd.DataFrame({"High": highs, "Low": lows}, index=index)


def test_barriers_use_volatility_and_economic_floor() -> None:
    barriers = create_barriers(
        current_price=10.0,
        volatility_estimate=0.1,
        lower_multiplier=0.5,
        upper_multiplier=1.0,
        horizon_key="60m",
        minimum_lower_distance=0.08,
    )

    assert barriers.lower_price == pytest.approx(9.92)
    assert barriers.upper_price == pytest.approx(10.1)


def test_lower_barrier_first_is_labeled_from_path_order() -> None:
    barriers = create_barriers(
        current_price=10.0,
        volatility_estimate=0.1,
        lower_multiplier=1.0,
        upper_multiplier=1.0,
        horizon_key="60m",
    )
    result = label_future_path(
        future_path([10.05, 10.06, 10.08], [9.95, 9.89, 9.92]),
        barriers=barriers,
        required_bars=3,
    )

    assert result.status is LabelStatus.VALID
    assert result.outcome is OutcomeClass.LOWER_FIRST
    assert result.time_to_event_bars == 2


def test_upper_barrier_first_is_labeled_from_path_order() -> None:
    barriers = create_barriers(
        current_price=10.0,
        volatility_estimate=0.1,
        lower_multiplier=1.0,
        upper_multiplier=1.0,
        horizon_key="60m",
    )
    result = label_future_path(
        future_path([10.04, 10.11, 10.08], [9.96, 9.95, 9.90]),
        barriers=barriers,
        required_bars=3,
    )

    assert result.outcome is OutcomeClass.UPPER_FIRST
    assert result.time_to_event_bars == 2


def test_neither_is_labeled_only_after_full_window() -> None:
    barriers = create_barriers(
        current_price=10.0,
        volatility_estimate=0.2,
        lower_multiplier=1.0,
        upper_multiplier=1.0,
        horizon_key="60m",
    )
    result = label_future_path(
        future_path([10.05, 10.06, 10.08], [9.95, 9.94, 9.92]),
        barriers=barriers,
        required_bars=3,
    )

    assert result.outcome is OutcomeClass.NEITHER
    assert result.time_to_event_bars == 3


def test_same_bar_double_touch_is_ambiguous_not_invented() -> None:
    barriers = create_barriers(
        current_price=10.0,
        volatility_estimate=0.1,
        lower_multiplier=1.0,
        upper_multiplier=1.0,
        horizon_key="60m",
    )
    result = label_future_path(
        future_path([10.11, 10.02], [9.89, 9.98]),
        barriers=barriers,
        required_bars=2,
    )

    assert result.status is LabelStatus.AMBIGUOUS
    assert result.outcome is None


def test_incomplete_path_is_not_labeled() -> None:
    barriers = create_barriers(
        current_price=10.0,
        volatility_estimate=0.1,
        lower_multiplier=1.0,
        upper_multiplier=1.0,
        horizon_key="60m",
    )
    result = label_future_path(
        future_path([10.02], [9.98]),
        barriers=barriers,
        required_bars=3,
    )

    assert result.status is LabelStatus.INSUFFICIENT_PATH
    assert result.outcome is None
