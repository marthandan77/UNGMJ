from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from ung_forecast.data.validator import DataValidationError, validate_ohlcv


def make_frame(*, duplicate: bool = False) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            "2026-07-10 14:30:00+00:00",
            "2026-07-10 14:35:00+00:00",
            "2026-07-10 14:40:00+00:00",
        ]
    )
    if duplicate:
        index = pd.DatetimeIndex([index[0], index[1], index[1]])
    return pd.DataFrame(
        {
            "Open": [10.0, 10.1, 10.2],
            "High": [10.2, 10.3, 10.4],
            "Low": [9.9, 10.0, 10.1],
            "Close": [10.1, 10.2, 10.3],
            "Volume": [1000, 1200, 900],
        },
        index=index,
    )


def test_completed_bar_filter_removes_open_bar() -> None:
    frame = make_frame()
    as_of = datetime(2026, 7, 10, 14, 43, tzinfo=UTC)

    validated = validate_ohlcv(frame, interval="5m", as_of=as_of)

    assert len(validated) == 2
    assert str(validated.index.tz) == "America/New_York"


def test_completed_bar_filter_keeps_closed_bar() -> None:
    frame = make_frame()
    as_of = datetime(2026, 7, 10, 14, 46, tzinfo=UTC)

    validated = validate_ohlcv(frame, interval="5m", as_of=as_of)

    assert len(validated) == 3


def test_duplicate_timestamps_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_ohlcv(
            make_frame(duplicate=True),
            interval="5m",
            as_of=datetime(2026, 7, 10, 15, 0, tzinfo=UTC),
        )


def test_missing_columns_are_rejected() -> None:
    frame = make_frame().drop(columns=["Volume"])
    with pytest.raises(DataValidationError, match="Missing required"):
        validate_ohlcv(
            frame,
            interval="5m",
            as_of=datetime(2026, 7, 10, 15, 0, tzinfo=UTC),
        )


def test_stale_data_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="stale"):
        validate_ohlcv(
            make_frame(),
            interval="5m",
            as_of=datetime(2026, 7, 10, 16, 0, tzinfo=UTC),
            stale_after=timedelta(minutes=20),
        )


def test_inconsistent_ohlc_is_rejected() -> None:
    frame = make_frame()
    frame.loc[frame.index[0], "High"] = 9.5
    with pytest.raises(DataValidationError, match="High is inconsistent"):
        validate_ohlcv(
            frame,
            interval="5m",
            as_of=datetime(2026, 7, 10, 15, 0, tzinfo=UTC),
        )
