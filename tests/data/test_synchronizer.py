from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from ung_forecast.data.schemas import DataProvenance, MarketDataBundle
from ung_forecast.data.synchronizer import SynchronizedMarketData
from ung_forecast.data.validator import DataValidationError


def bundle(symbol: str, timestamps: list[str], interval: str = "5m") -> MarketDataBundle:
    frame = pd.DataFrame(
        {
            "Open": [10.0 + i for i in range(len(timestamps))],
            "High": [10.2 + i for i in range(len(timestamps))],
            "Low": [9.8 + i for i in range(len(timestamps))],
            "Close": [10.1 + i for i in range(len(timestamps))],
            "Volume": [1000 + i for i in range(len(timestamps))],
        },
        index=pd.DatetimeIndex(timestamps),
    )
    return MarketDataBundle(
        frame=frame,
        provenance=DataProvenance(
            provider="test",
            symbol=symbol,
            interval=interval,
            downloaded_at=datetime(2026, 7, 10, 15, 0, tzinfo=UTC),
            source_timezone="UTC",
            normalized_timezone="America/New_York",
            adjusted_prices=False,
        ),
    )


def test_alignment_uses_exact_intersection_only() -> None:
    primary = bundle(
        "UNG",
        [
            "2026-07-10 10:30:00-04:00",
            "2026-07-10 10:35:00-04:00",
            "2026-07-10 10:40:00-04:00",
        ],
    )
    comparison = bundle(
        "NG=F",
        [
            "2026-07-10 10:35:00-04:00",
            "2026-07-10 10:40:00-04:00",
            "2026-07-10 10:45:00-04:00",
        ],
    )

    aligned_primary, aligned_comparison = SynchronizedMarketData(primary, comparison).align()

    assert list(aligned_primary.index) == list(aligned_comparison.index)
    assert len(aligned_primary) == 2
    assert aligned_primary.index[0] == pd.Timestamp("2026-07-10 10:35:00-04:00")


def test_alignment_rejects_interval_mismatch() -> None:
    with pytest.raises(DataValidationError, match="Intervals must match"):
        SynchronizedMarketData(
            bundle("UNG", ["2026-07-10 10:30:00-04:00"], interval="5m"),
            bundle("NG=F", ["2026-07-10 10:30:00-04:00"], interval="15m"),
        )


def test_alignment_rejects_no_overlap() -> None:
    primary = bundle("UNG", ["2026-07-10 10:30:00-04:00"])
    comparison = bundle("NG=F", ["2026-07-10 10:35:00-04:00"])

    with pytest.raises(DataValidationError, match="No synchronized"):
        SynchronizedMarketData(primary, comparison).align()
