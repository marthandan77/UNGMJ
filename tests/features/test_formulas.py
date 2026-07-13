from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ung_forecast.features.formulas import (
    log_return,
    price_scaled_volatility,
    realized_volatility,
    session_vwap,
    standardized_vwap_deviation,
    time_of_day_encoding,
    variance_ratio,
)


def frame() -> pd.DataFrame:
    index = pd.date_range("2026-07-10 09:30", periods=8, freq="5min", tz="America/New_York")
    close = pd.Series([10.0, 10.1, 10.2, 10.1, 10.3, 10.4, 10.5, 10.6], index=index)
    return pd.DataFrame(
        {
            "Open": close - 0.05,
            "High": close + 0.10,
            "Low": close - 0.10,
            "Close": close,
            "Volume": [100, 200, 150, 250, 300, 200, 400, 350],
        },
        index=index,
    )


def test_log_return_matches_definition() -> None:
    data = frame()
    result = log_return(data["Close"], 2)
    expected = math.log(10.2 / 10.0)
    assert result.iloc[2] == pytest.approx(expected)
    assert result.iloc[:2].isna().all()


def test_realized_volatility_is_root_sum_squared_returns() -> None:
    data = frame()
    result = realized_volatility(data["Close"], 3)
    one_period = np.log(data["Close"] / data["Close"].shift(1))
    expected = math.sqrt(float(one_period.iloc[1:4].pow(2).sum()))
    assert result.iloc[3] == pytest.approx(expected)


def test_price_scaled_volatility_has_price_units() -> None:
    data = frame()
    relative = realized_volatility(data["Close"], 3)
    result = price_scaled_volatility(data["Close"], relative)
    assert result.iloc[3] == pytest.approx(data["Close"].iloc[3] * relative.iloc[3])
    assert result.name == "price_scaled_volatility"


def test_price_scaled_volatility_requires_exact_alignment() -> None:
    data = frame()
    relative = realized_volatility(data["Close"], 3).iloc[1:]
    with pytest.raises(ValueError, match="align exactly"):
        price_scaled_volatility(data["Close"], relative)


def test_session_vwap_uses_only_current_and_prior_rows() -> None:
    data = frame()
    result = session_vwap(data)
    typical = (data["High"] + data["Low"] + data["Close"]) / 3.0
    expected_second = float(
        (typical.iloc[0] * 100 + typical.iloc[1] * 200) / (100 + 200)
    )
    assert result.iloc[1] == pytest.approx(expected_second)

    changed_future = data.copy()
    changed_future.iloc[-1, changed_future.columns.get_loc("Close")] = 99.0
    changed_future.iloc[-1, changed_future.columns.get_loc("High")] = 99.1
    changed_future_result = session_vwap(changed_future)
    pd.testing.assert_series_equal(result.iloc[:-1], changed_future_result.iloc[:-1])


def test_standardized_vwap_deviation_is_right_aligned() -> None:
    data = frame()
    original = standardized_vwap_deviation(data, window=3)
    changed = data.copy()
    changed.iloc[-1, changed.columns.get_loc("Close")] = 50.0
    changed.iloc[-1, changed.columns.get_loc("High")] = 50.1
    revised = standardized_vwap_deviation(changed, window=3)
    pd.testing.assert_series_equal(original.iloc[:-1], revised.iloc[:-1])


def test_variance_ratio_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="window must exceed q"):
        variance_ratio(frame()["Close"], q=4, window=4)


def test_time_encoding_is_bounded() -> None:
    encoded = time_of_day_encoding(frame().index)
    assert encoded["time_sin"].between(-1.0, 1.0).all()
    assert encoded["time_cos"].between(-1.0, 1.0).all()
