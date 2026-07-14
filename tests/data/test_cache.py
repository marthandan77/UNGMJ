from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ung_forecast.data.cache import ParquetCache
from ung_forecast.data.schemas import DataProvenance, MarketDataBundle


def test_parquet_cache_round_trip(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "Open": [10.0, 10.1],
            "High": [10.2, 10.3],
            "Low": [9.9, 10.0],
            "Close": [10.1, 10.2],
            "Volume": [1000, 1200],
        },
        index=pd.DatetimeIndex(
            ["2026-07-10 10:30:00-04:00", "2026-07-10 10:35:00-04:00"]
        ),
    )
    bundle = MarketDataBundle(
        frame=frame,
        provenance=DataProvenance(
            provider="yfinance",
            symbol="NG=F",
            interval="5m",
            downloaded_at=datetime(2026, 7, 10, 15, 0, tzinfo=UTC),
            source_timezone="UTC",
            normalized_timezone="America/New_York",
            adjusted_prices=False,
        ),
    )

    cache = ParquetCache(tmp_path)
    written = cache.write(bundle)
    loaded = cache.read("NG=F", "5m")

    assert written.provenance.cache_path is not None
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded.frame, frame, check_freq=False)
    assert loaded.provenance.symbol == "NG=F"
    assert loaded.provenance.cache_path == written.provenance.cache_path


def test_missing_cache_returns_none(tmp_path) -> None:
    assert ParquetCache(tmp_path).read("UNG", "5m") is None
