"""Twelve Data time-series adapter for Streamlit runtime market data.

The adapter converts provider-specific JSON into the project's canonical OHLCV
schema before the shared validator, cache, feature, and model layers see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import pandas as pd
import requests

from .schemas import DataProvenance, MarketDataBundle
from .validator import validate_ohlcv


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> dict[str, Any]: ...


class HttpClient(Protocol):
    def get(
        self,
        url: str,
        *,
        params: dict[str, Any],
        timeout: float,
    ) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class TwelveDataCredentials:
    api_key: str

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("Twelve Data api_key is required")


INTERVAL_MAP: dict[str, str] = {
    "5m": "5min",
    "15m": "15min",
    "1d": "1day",
}

OUTPUT_SIZE_BY_INTERVAL: dict[str, int] = {
    "5m": 5000,
    "15m": 5000,
    "1d": 5000,
}


class TwelveDataMarketDataProvider:
    def __init__(
        self,
        credentials: TwelveDataCredentials,
        *,
        base_url: str = "https://api.twelvedata.com",
        timezone: str = "America/New_York",
        adjusted_prices: bool = False,
        http_client: HttpClient | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self.timezone = timezone
        self.adjusted_prices = adjusted_prices
        self.http_client = http_client or requests.Session()
        self.timeout = timeout

    @staticmethod
    def _stale_after(interval: str) -> timedelta:
        mapping = {
            "5m": timedelta(minutes=20),
            "15m": timedelta(minutes=45),
            "1d": timedelta(days=4),
        }
        try:
            return mapping[interval]
        except KeyError as exc:
            raise ValueError(f"Twelve Data interval is not supported: {interval}") from exc

    def download(
        self,
        *,
        symbol: str,
        interval: str,
        period: str = "",
        as_of: datetime | None = None,
    ) -> MarketDataBundle:
        del period
        effective_as_of = as_of or datetime.now(UTC)
        if effective_as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        try:
            provider_interval = INTERVAL_MAP[interval]
            output_size = OUTPUT_SIZE_BY_INTERVAL[interval]
        except KeyError as exc:
            raise ValueError(f"Twelve Data interval is not supported: {interval}") from exc

        response = self.http_client.get(
            f"{self.base_url}/time_series",
            params={
                "symbol": symbol,
                "interval": provider_interval,
                "outputsize": output_size,
                "timezone": self.timezone,
                "format": "JSON",
                "apikey": self.credentials.api_key,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "error" or "values" not in payload:
            code = payload.get("code", "unknown")
            message = payload.get("message", "Twelve Data returned an invalid response")
            raise RuntimeError(f"Twelve Data error {code}: {message}")

        values = payload.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Twelve Data returned no candle data for {symbol} at {interval}")
        frame = pd.DataFrame(values)
        required = {"datetime", "open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Twelve Data response is missing fields: {sorted(missing)}")

        frame = frame.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        timestamps = pd.to_datetime(frame.pop("datetime"), errors="raise")
        if not isinstance(timestamps, pd.Series):
            raise ValueError("Twelve Data timestamps could not be parsed")
        index = pd.DatetimeIndex(timestamps)
        if index.tz is None:
            index = index.tz_localize(self.timezone, ambiguous="raise", nonexistent="raise")
        else:
            index = index.tz_convert(self.timezone)
        frame.index = index
        frame = frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame = frame.sort_index()

        validated = validate_ohlcv(
            frame,
            interval=interval,
            as_of=effective_as_of,
            timezone=self.timezone,
            stale_after=self._stale_after(interval),
        )
        provenance = DataProvenance(
            provider="twelvedata",
            symbol=symbol,
            interval=interval,
            downloaded_at=datetime.now(UTC),
            source_timezone=self.timezone,
            normalized_timezone=self.timezone,
            adjusted_prices=self.adjusted_prices,
        )
        return MarketDataBundle(frame=validated, provenance=provenance)
