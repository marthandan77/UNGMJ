"""Conservative guard against issuing forecasts after the target session has started."""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from .contracts import DataValidationError

NEW_YORK = ZoneInfo("America/New_York")
REGULAR_SESSION_OPEN = time(9, 30)


def validate_forecast_timing(origin_session: date, now: datetime) -> None:
    """Fail closed if the current NY weekday session has started beyond the data origin."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise DataValidationError("forecast cutoff must be timezone-aware")
    new_york_now = now.astimezone(NEW_YORK)
    if (
        new_york_now.weekday() < 5
        and new_york_now.time().replace(tzinfo=None) >= REGULAR_SESSION_OPEN
        and origin_session < new_york_now.date()
    ):
        raise DataValidationError(
            f"Forecast origin {origin_session} is stale for the current New York session "
            f"({new_york_now.date()} at {new_york_now:%H:%M} ET); the target session has "
            "already started. Wait until the next pre-open scheduled run."
        )
