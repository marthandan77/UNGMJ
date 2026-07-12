from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ung_forecast.data.cache import ParquetCache
from ung_forecast.data.runtime_loader import REQUIRED_INTERVALS, load_runtime_market_data
from ung_forecast.data.schemas import DataProvenance, MarketDataBundle
from ung_forecast.horizons import HorizonKey


class FakeProvider:
    def __init__(self, *, fail_interval: str | None = None) -> None:
        self.fail_interval = fail_interval
        self.calls: list[str] = []

    def download(
        self,
        *,
        symbol: str,
        interval: str,
        period: str,
        as_of: datetime,
    ) -> MarketDataBundle:
        del period, as_of
        self.calls.append(interval)
        if interval == self.fail_interval:
            raise RuntimeError("provider unavailable")
        index = pd.date_range("2026-07-10 10:30", periods=2, freq="5min", tz="America/New_York")
        frame = pd.DataFrame(
            {
                "Open": [10.0, 10.1],
                "High": [10.2, 10.3],
                "Low": [9.9, 10.0],
                "Close": [10.1, 10.2],
                "Volume": [1000, 1100],
            },
            index=index,
        )
        return MarketDataBundle(
            frame=frame,
            provenance=DataProvenance(
                provider="fake",
                symbol=symbol,
                interval=interval,
                downloaded_at=datetime(2026, 7, 10, 15, 0, tzinfo=UTC),
                source_timezone="America/New_York",
                normalized_timezone="America/New_York",
                adjusted_prices=False,
            ),
        )


def test_loader_fetches_each_required_interval_once(tmp_path) -> None:
    provider = FakeProvider()
    result = load_runtime_market_data(
        provider,
        ParquetCache(tmp_path),
        symbol="UNG",
        as_of=datetime(2026, 7, 10, 15, 0, tzinfo=UTC),
    )

    assert tuple(provider.calls) == REQUIRED_INTERVALS
    assert set(result.bundles_by_interval) == set(REQUIRED_INTERVALS)
    assert set(result.source_by_interval.values()) == {"live"}
    assert result.bundle_for_horizon(HorizonKey.DAY_1).provenance.interval == "1d"


def test_loader_uses_explicit_cache_fallback(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    initial = load_runtime_market_data(
        FakeProvider(),
        cache,
        symbol="UNG",
        as_of=datetime(2026, 7, 10, 15, 0, tzinfo=UTC),
    )
    assert initial.source_by_interval["15m"] == "live"

    result = load_runtime_market_data(
        FakeProvider(fail_interval="15m"),
        cache,
        symbol="UNG",
        as_of=datetime(2026, 7, 10, 15, 5, tzinfo=UTC),
    )

    assert result.source_by_interval["15m"] == "cache_fallback"
    assert "provider unavailable" in result.errors_by_interval["15m"]


def test_loader_rejects_stale_cache_fallback(tmp_path) -> None:
    cache = ParquetCache(tmp_path)
    load_runtime_market_data(
        FakeProvider(),
        cache,
        symbol="UNG",
        as_of=datetime(2026, 7, 10, 15, 0, tzinfo=UTC),
    )
    result = load_runtime_market_data(
        FakeProvider(fail_interval="5m"),
        cache,
        symbol="UNG",
        as_of=datetime(2026, 7, 10, 16, 0, tzinfo=UTC),
    )
    assert "5m" not in result.bundles_by_interval
    assert "cache rejected" in result.errors_by_interval["5m"]


def test_loader_does_not_hide_failure_without_cache_fallback(tmp_path) -> None:
    result = load_runtime_market_data(
        FakeProvider(fail_interval="5m"),
        ParquetCache(tmp_path),
        symbol="UNG",
        as_of=datetime(2026, 7, 10, 15, 0, tzinfo=UTC),
        allow_cache_fallback=False,
    )

    assert "5m" not in result.bundles_by_interval
    assert "5m" in result.errors_by_interval
