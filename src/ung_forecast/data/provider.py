"""Provider-neutral market-data interface and initial yfinance implementation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

import pandas as pd
import yfinance as yf

from .schemas import DataProvenance, MarketDataBundle
from .validator import validate_ohlcv


class MarketDataProvider(Protocol):
    def download(
        self,
        *,
        symbol: str,
        interval: str,
        period: str,
        as_of: datetime,
    ) -> MarketDataBundle: ...


class YFinanceProvider:
    def __init__(
        self,
        *,
        timezone: str = "America/New_York",
        adjusted_prices: bool = False,
    ) -> None:
        self.timezone = timezone
        self.adjusted_prices = adjusted_prices

    @staticmethod
    def _default_stale_after(interval: str) -> timedelta:
        mapping = {
            "5m": timedelta(minutes=20),
            "15m": timedelta(minutes=45),
            "1h": timedelta(hours=3),
            "1d": timedelta(days=4),
        }
        try:
            return mapping[interval]
        except KeyError as exc:
            raise ValueError(f"Unsupported interval: {interval}") from exc

    @staticmethod
    def _flatten_columns(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if not isinstance(frame.columns, pd.MultiIndex):
            return frame

        result = frame.copy()
        if symbol in result.columns.get_level_values(-1):
            result = cast(
                pd.DataFrame,
                result.xs(symbol, axis=1, level=-1, drop_level=True),
            )
        elif symbol in result.columns.get_level_values(0):
            result = cast(
                pd.DataFrame,
                result.xs(symbol, axis=1, level=0, drop_level=True),
            )
        else:
            raise ValueError(f"Unable to identify symbol columns for {symbol}")
        return result

    def _download_frame(self, *, symbol: str, interval: str, period: str) -> pd.DataFrame:
        raw = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=self.adjusted_prices,
            progress=False,
            threads=False,
            group_by="column",
        )
        if raw.empty:
            raise ValueError(f"yfinance returned no data for {symbol} at {interval}")
        flattened = self._flatten_columns(raw, symbol)
        if not isinstance(flattened.index, pd.DatetimeIndex):
            raise ValueError("yfinance data must use a DatetimeIndex")
        return flattened

    def _bundle(
        self,
        frame: pd.DataFrame,
        *,
        symbol: str,
        interval: str,
        as_of: datetime,
        stale_after: timedelta,
    ) -> MarketDataBundle:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        index = frame.index
        source_timezone = str(index.tz) if index.tz is not None else "naive"
        normalized = frame.copy()
        if index.tz is None:
            normalized.index = index.tz_localize(self.timezone)

        validated = validate_ohlcv(
            normalized,
            interval=interval,
            as_of=as_of,
            timezone=self.timezone,
            stale_after=stale_after,
        )
        provenance = DataProvenance(
            provider="yfinance",
            symbol=symbol,
            interval=interval,
            downloaded_at=datetime.now(UTC),
            source_timezone=source_timezone,
            normalized_timezone=self.timezone,
            adjusted_prices=self.adjusted_prices,
        )
        return MarketDataBundle(frame=validated, provenance=provenance)

    def download(
        self,
        *,
        symbol: str,
        interval: str,
        period: str,
        as_of: datetime | None = None,
    ) -> MarketDataBundle:
        effective_as_of = as_of or datetime.now(UTC)
        frame = self._download_frame(symbol=symbol, interval=interval, period=period)
        return self._bundle(
            frame,
            symbol=symbol,
            interval=interval,
            as_of=effective_as_of,
            stale_after=self._default_stale_after(interval),
        )

    def download_historical(
        self,
        *,
        symbol: str,
        interval: str,
        period: str,
    ) -> MarketDataBundle:
        """Download historical data without applying live-feed freshness rejection."""

        frame = self._download_frame(symbol=symbol, interval=interval, period=period)
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.empty:
            raise ValueError("Historical yfinance data requires a non-empty DatetimeIndex")
        latest = pd.Timestamp(frame.index[-1])
        if latest.tzinfo is None:
            latest = latest.tz_localize(self.timezone)
        as_of = latest.to_pydatetime() + timedelta(seconds=1)
        return self._bundle(
            frame,
            symbol=symbol,
            interval=interval,
            as_of=as_of,
            stale_after=timedelta(days=1),
        )
