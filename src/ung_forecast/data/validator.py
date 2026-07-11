"""Validation rules for market data used by every horizon."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
INTERVAL_TO_DELTA = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


class DataValidationError(ValueError):
    """Raised when market data violates a non-negotiable data contract."""


def _datetime_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError("Market data index must be a DatetimeIndex")
    if frame.index.tz is None:
        raise DataValidationError("Market data timestamps must be timezone-aware")
    return frame.index


def normalize_timezone(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    index = _datetime_index(frame)
    normalized = frame.copy()
    normalized.index = index.tz_convert(ZoneInfo(timezone))
    return normalized


def remove_incomplete_last_bar(
    frame: pd.DataFrame,
    *,
    interval: str,
    as_of: datetime,
) -> pd.DataFrame:
    index = _datetime_index(frame)
    if interval not in INTERVAL_TO_DELTA:
        raise DataValidationError(f"Unsupported interval: {interval}")
    if as_of.tzinfo is None:
        raise DataValidationError("as_of must be timezone-aware")
    if frame.empty:
        raise DataValidationError("Market data frame cannot be empty")

    interval_delta = INTERVAL_TO_DELTA[interval]
    last_start = index[-1].to_pydatetime()
    if last_start + interval_delta > as_of:
        return frame.iloc[:-1].copy()
    return frame.copy()


def validate_ohlcv(
    frame: pd.DataFrame,
    *,
    interval: str,
    as_of: datetime,
    timezone: str = "America/New_York",
    stale_after: timedelta | None = None,
) -> pd.DataFrame:
    """Return normalized, completed, validated OHLCV data.

    The function never forward-fills and never silently removes duplicate timestamps.
    """

    index = _datetime_index(frame)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required OHLCV columns: {missing}")
    if index.has_duplicates:
        raise DataValidationError("Duplicate timestamps detected")
    if not index.is_monotonic_increasing:
        raise DataValidationError("Timestamps must be strictly increasing")

    numeric = frame.loc[:, REQUIRED_COLUMNS]
    if numeric.isna().any().any():
        raise DataValidationError("OHLCV data contains missing values")
    if (numeric[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise DataValidationError("Prices must be strictly positive")
    if (numeric["Volume"] < 0).any():
        raise DataValidationError("Volume cannot be negative")
    if (numeric["High"] < numeric[["Open", "Close", "Low"]].max(axis=1)).any():
        raise DataValidationError("High is inconsistent with OHLC values")
    if (numeric["Low"] > numeric[["Open", "Close", "High"]].min(axis=1)).any():
        raise DataValidationError("Low is inconsistent with OHLC values")

    normalized = normalize_timezone(frame, timezone)
    completed = remove_incomplete_last_bar(normalized, interval=interval, as_of=as_of)
    if completed.empty:
        raise DataValidationError("No completed bars remain after filtering")

    completed_index = _datetime_index(completed)
    if stale_after is not None:
        latest = completed_index[-1].to_pydatetime()
        if as_of - latest > stale_after:
            raise DataValidationError("Market data is stale")

    return completed
