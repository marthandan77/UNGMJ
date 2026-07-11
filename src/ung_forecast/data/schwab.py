"""Schwab market-data adapter for the GitHub + Streamlit deployment.

Endpoint URLs are configurable because Schwab's developer portal is the source
of truth. Secrets are injected at runtime and are never stored in the repo.
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
        headers: dict[str, str],
        params: dict[str, Any],
        timeout: float,
    ) -> HttpResponse: ...

    def post(
        self,
        url: str,
        *,
        data: dict[str, str],
        auth: tuple[str, str],
        timeout: float,
    ) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class SchwabCredentials:
    client_id: str
    client_secret: str
    refresh_token: str | None = None
    access_token: str | None = None
    token_url: str = "https://api.schwabapi.com/v1/oauth/token"

    def __post_init__(self) -> None:
        if not self.client_id or not self.client_secret:
            raise ValueError("Schwab client_id and client_secret are required")
        if not self.access_token and not self.refresh_token:
            raise ValueError("Provide either a Schwab access_token or refresh_token")


class SchwabTokenProvider:
    def __init__(
        self,
        credentials: SchwabCredentials,
        *,
        http_client: HttpClient | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.credentials = credentials
        self.http_client = http_client or requests.Session()
        self.timeout = timeout
        self._access_token = credentials.access_token
        self._expires_at: datetime | None = None

    def get_access_token(self, *, now: datetime | None = None) -> str:
        effective_now = now or datetime.now(UTC)
        if self._access_token and (
            self._expires_at is None or effective_now < self._expires_at - timedelta(seconds=30)
        ):
            return self._access_token
        if not self.credentials.refresh_token:
            raise RuntimeError("Schwab access token expired and no refresh token is configured")

        response = self.http_client.post(
            self.credentials.token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self.credentials.refresh_token,
            },
            auth=(self.credentials.client_id, self.credentials.client_secret),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        token = str(payload.get("access_token", ""))
        expires_in = int(payload.get("expires_in", 0))
        if not token or expires_in <= 0:
            raise RuntimeError("Schwab token response is missing access_token or expires_in")
        self._access_token = token
        self._expires_at = effective_now + timedelta(seconds=expires_in)
        return token


@dataclass(frozen=True, slots=True)
class SchwabPriceHistoryRequest:
    period_type: str
    period: int
    frequency_type: str
    frequency: int


INTERVAL_REQUESTS: dict[str, SchwabPriceHistoryRequest] = {
    "5m": SchwabPriceHistoryRequest("day", 10, "minute", 5),
    "15m": SchwabPriceHistoryRequest("day", 10, "minute", 15),
    "1d": SchwabPriceHistoryRequest("year", 10, "daily", 1),
}


class SchwabMarketDataProvider:
    def __init__(
        self,
        token_provider: SchwabTokenProvider,
        *,
        base_url: str = "https://api.schwabapi.com/marketdata/v1",
        timezone: str = "America/New_York",
        http_client: HttpClient | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.token_provider = token_provider
        self.base_url = base_url.rstrip("/")
        self.timezone = timezone
        self.http_client = http_client or requests.Session()
        self.timeout = timeout

    @staticmethod
    def _stale_after(interval: str) -> timedelta:
        return {
            "5m": timedelta(minutes=20),
            "15m": timedelta(minutes=45),
            "1d": timedelta(days=4),
        }[interval]

    def download(
        self,
        *,
        symbol: str,
        interval: str,
        period: str = "",
        as_of: datetime | None = None,
    ) -> MarketDataBundle:
        del period  # Schwab uses explicit periodType/period parameters.
        effective_as_of = as_of or datetime.now(UTC)
        if effective_as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        try:
            request = INTERVAL_REQUESTS[interval]
        except KeyError as exc:
            raise ValueError(f"Schwab interval is not supported: {interval}") from exc

        token = self.token_provider.get_access_token(now=effective_as_of)
        response = self.http_client.get(
            f"{self.base_url}/pricehistory",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            params={
                "symbol": symbol,
                "periodType": request.period_type,
                "period": request.period,
                "frequencyType": request.frequency_type,
                "frequency": request.frequency,
                "needExtendedHoursData": "false",
                "needPreviousClose": "true",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        candles = payload.get("candles")
        if not isinstance(candles, list) or not candles:
            raise ValueError(f"Schwab returned no candle data for {symbol} at {interval}")

        frame = pd.DataFrame(candles)
        required = {"open", "high", "low", "close", "volume", "datetime"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Schwab candle response is missing fields: {sorted(missing)}")
        frame = frame.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        frame.index = pd.to_datetime(frame.pop("datetime"), unit="ms", utc=True)
        frame = frame.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
        validated = validate_ohlcv(
            frame,
            interval=interval,
            as_of=effective_as_of,
            timezone=self.timezone,
            stale_after=self._stale_after(interval),
        )
        provenance = DataProvenance(
            provider="schwab",
            symbol=symbol,
            interval=interval,
            downloaded_at=datetime.now(UTC),
            source_timezone="UTC",
            normalized_timezone=self.timezone,
            adjusted_prices=False,
        )
        return MarketDataBundle(frame=validated, provenance=provenance)
