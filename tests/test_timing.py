from datetime import date, datetime, timezone

import pytest

from ungmj.contracts import DataValidationError
from ungmj.timing import validate_forecast_timing


def test_blocks_forecast_after_target_session_has_started() -> None:
    # 10:38 ET on Friday; the most recent completed close is Thursday.
    now = datetime(2026, 10, 16, 14, 38, tzinfo=timezone.utc)
    with pytest.raises(DataValidationError, match="target session has already started"):
        validate_forecast_timing(date(2026, 10, 15), now)


def test_allows_preopen_forecast_from_previous_completed_session() -> None:
    # 22:30 ET Monday; next session has not opened and Monday's close is available.
    now = datetime(2026, 10, 20, 2, 30, tzinfo=timezone.utc)
    validate_forecast_timing(date(2026, 10, 19), now)


def test_requires_timezone_aware_cutoff() -> None:
    with pytest.raises(DataValidationError, match="timezone-aware"):
        validate_forecast_timing(date(2026, 10, 15), datetime(2026, 10, 16, 14, 38))
