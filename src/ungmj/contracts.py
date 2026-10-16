"""Core domain contracts and fail-closed validation for market observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Final, Literal

import numpy as np
import pandas as pd

HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 5)
OHLCV_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
ForecastStatus = Literal["READY", "PROVISIONAL", "BLOCKED"]


class DataValidationError(ValueError):
    """Raised when source data cannot safely support downstream research."""


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Provenance for one downloaded dataset; timestamps must be timezone-aware."""

    provider: str
    instrument: str
    retrieved_at: datetime
    published_through: date | None
    content_sha256: str
    source_url: str

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be blank")
        if not self.instrument.strip():
            raise ValueError("instrument must not be blank")
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        _validate_sha256(self.content_sha256, "content_sha256")
        if not self.source_url.startswith(("https://", "http://")):
            raise ValueError("source_url must be an HTTP(S) URL")


@dataclass(frozen=True, slots=True)
class ForecastVintage:
    """Immutable record of a forecast issued before its target outcome is known."""

    forecast_id: str
    instrument: str
    as_of: datetime
    origin_session: date
    horizon_sessions: int
    reference_close: float
    probability_up: float
    expected_return: float
    expected_close: float
    predicted_low: float
    predicted_high: float
    lower_return_quantile: float
    upper_return_quantile: float
    calibration_sample_count: int
    calibration_method: str
    model_name: str
    model_version: str
    input_fingerprint: str
    status: ForecastStatus = "READY"

    def __post_init__(self) -> None:
        if not self.forecast_id.strip():
            raise ValueError("forecast_id must not be blank")
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        if not isinstance(self.origin_session, date):
            raise ValueError("origin_session must be a date")
        if self.horizon_sessions not in HORIZONS:
            raise ValueError(f"horizon_sessions must be one of {HORIZONS}")
        if self.calibration_sample_count < 0 or not self.calibration_method.strip():
            raise ValueError("calibration metadata must be valid")
        if not self.instrument.strip() or not self.model_name.strip() or not self.model_version.strip():
            raise ValueError("instrument and model identity fields must not be blank")
        numeric = (
            self.reference_close, self.probability_up, self.expected_return, self.expected_close,
            self.predicted_low, self.predicted_high, self.lower_return_quantile,
            self.upper_return_quantile,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("forecast numeric fields must all be finite")
        if self.reference_close <= 0 or self.expected_close <= 0:
            raise ValueError("reference_close and expected_close must be positive")
        if not 0 <= self.probability_up <= 1:
            raise ValueError("probability_up must be between 0 and 1")
        if self.predicted_low <= 0 or self.predicted_high <= 0:
            raise ValueError("predicted range bounds must be positive")
        if self.predicted_low > self.predicted_high:
            raise ValueError("predicted_low must not exceed predicted_high")
        if self.lower_return_quantile > self.upper_return_quantile:
            raise ValueError("lower_return_quantile must not exceed upper_return_quantile")
        _validate_sha256(self.input_fingerprint, "input_fingerprint")


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be hexadecimal") from exc


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp suitable for provenance records."""
    return datetime.now(timezone.utc)


def validate_ohlcv(
    frame: pd.DataFrame,
    *,
    expected_latest_session: date | None = None,
    instrument: str = "UNG",
) -> pd.DataFrame:
    """Validate daily OHLCV bars and return a defensive, date-sorted copy.

    Freshness is not inferred from wall-clock time: callers must supply an independently
    justified expected session when freshness is required. The validator never silently
    fills, deduplicates, resamples, or repairs bad observations.
    """
    if frame.empty:
        raise DataValidationError(f"{instrument}: OHLCV dataset is empty")
    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"{instrument}: missing required columns: {missing}")
    result = frame.loc[:, list(OHLCV_COLUMNS)].copy()
    try:
        parsed_dates = pd.to_datetime(frame.index, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError("index must contain valid session dates") from exc
    if getattr(parsed_dates, "tz", None) is not None:
        parsed_dates = parsed_dates.tz_localize(None)
    normalized_dates = pd.DatetimeIndex(parsed_dates).normalize()
    if normalized_dates.has_duplicates:
        raise DataValidationError(f"{instrument}: duplicate daily sessions detected")
    result.index = normalized_dates
    result.index.name = "session"
    try:
        for column in OHLCV_COLUMNS:
            result[column] = pd.to_numeric(result[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError("OHLCV values must be numeric") from exc
    values = result.loc[:, list(OHLCV_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataValidationError("OHLCV values must not contain NaN or infinity")
    prices = result[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise DataValidationError("OHLC prices must all be positive")
    if (result["volume"] < 0).any():
        raise DataValidationError("volume must be non-negative")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any():
        raise DataValidationError("high must be >= open, close, and low")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise DataValidationError("low must be <= open, close, and high")
    result = result.sort_index()
    if expected_latest_session is not None:
        expected = pd.Timestamp(expected_latest_session).normalize()
        actual = result.index[-1]
        if actual != expected:
            raise DataValidationError(
                f"{instrument}: latest session {actual.date()} does not match "
                f"expected session {expected.date()}"
            )
    return result
