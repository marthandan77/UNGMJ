"""Load and cache the market-data intervals required by all forecast horizons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ung_forecast.horizons import HORIZON_SPECS, HorizonKey

from .cache import ParquetCache
from .provider import MarketDataProvider
from .schemas import MarketDataBundle


@dataclass(frozen=True, slots=True)
class RuntimeDataSet:
    bundles_by_interval: dict[str, MarketDataBundle]
    source_by_interval: dict[str, str]
    errors_by_interval: dict[str, str]

    def bundle_for_horizon(self, horizon: HorizonKey) -> MarketDataBundle:
        interval = HORIZON_SPECS[horizon].source_interval
        try:
            return self.bundles_by_interval[interval]
        except KeyError as exc:
            raise KeyError(f"No runtime data is available for horizon {horizon.value}") from exc


REQUIRED_INTERVALS = tuple(
    dict.fromkeys(specification.source_interval for specification in HORIZON_SPECS.values())
)


def load_runtime_market_data(
    provider: MarketDataProvider,
    cache: ParquetCache,
    *,
    symbol: str,
    as_of: datetime,
    allow_cache_fallback: bool = True,
) -> RuntimeDataSet:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")

    bundles: dict[str, MarketDataBundle] = {}
    sources: dict[str, str] = {}
    errors: dict[str, str] = {}

    for interval in REQUIRED_INTERVALS:
        try:
            live_bundle = provider.download(
                symbol=symbol,
                interval=interval,
                period="",
                as_of=as_of,
            )
            cached_bundle = cache.write(live_bundle)
        except Exception as exc:
            errors[interval] = f"{type(exc).__name__}: {exc}"
            if not allow_cache_fallback:
                continue
            cached_bundle = cache.read(symbol, interval)
            if cached_bundle is None:
                continue
            bundles[interval] = cached_bundle
            sources[interval] = "cache_fallback"
        else:
            bundles[interval] = cached_bundle
            sources[interval] = "live"

    return RuntimeDataSet(
        bundles_by_interval=bundles,
        source_by_interval=sources,
        errors_by_interval=errors,
    )
