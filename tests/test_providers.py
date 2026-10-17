import json

import pandas as pd
import pytest

from ung_forecast.contracts import DataValidationError, FetchedFrame, SourceSnapshot
from ung_forecast.providers import compare_sources
from datetime import datetime, timezone


def frame(provider: str, multiplier: float = 1.0) -> FetchedFrame:
    index = pd.date_range("2024-01-01", periods=600, freq="B")
    values = pd.DataFrame({"close": [10.0 * multiplier] * 600}, index=index)
    snap = SourceSnapshot(provider, "UNG", datetime.now(timezone.utc), index[-1].date(), "a" * 64, "https://example.com")
    return FetchedFrame(values, snap, b"fixture")


def test_independent_sources_compare() -> None:
    assert "confirmed" in compare_sources(frame("nasdaq"), frame("yahoo"))


def test_same_provider_is_rejected() -> None:
    with pytest.raises(DataValidationError):
        compare_sources(frame("nasdaq"), frame("nasdaq"))
