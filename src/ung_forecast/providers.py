"""Alpaca market-data adapters for UNG daily and extended-session bars."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .contracts import DataValidationError, FetchedFrame, SourceSnapshot, validate_ohlcv


ET = ZoneInfo("America/New_York")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/UNG/bars"


def _clean_secret(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _streamlit_credential_pair() -> tuple[str, str] | None:
    try:
        import streamlit as st
        secrets = st.secrets
        if not isinstance(secrets, Mapping):
            return None

        # Look up a matched pair in one location. Arbitrary recursive matches
        # can accidentally combine unrelated credentials from other tables.
        aliases = (
            ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
            ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"),
            ("ALPACA_API_KEY", "ALPACA_API_SECRET"),
            ("API_KEY_ID", "API_SECRET_KEY"),
            ("API_KEY", "API_SECRET"),
            ("key_id", "secret_key"),
            ("key", "secret"),
        )

        def pair_from(table: Mapping) -> tuple[str, str] | None:
            for key_name, secret_name in aliases:
                key = _clean_secret(table.get(key_name))
                secret = _clean_secret(table.get(secret_name))
                if key and secret:
                    return key, secret
            return None

        # Root credentials are the standard Streamlit secrets.toml layout.
        pair = pair_from(secrets)
        if pair:
            return pair
        # Also accept the common dedicated [alpaca] table layout.
        for section_name in ("alpaca", "ALPACA"):
            section = secrets.get(section_name)
            if isinstance(section, Mapping):
                pair = pair_from(section)
                if pair:
                    return pair
    except Exception:
        # Streamlit secrets are unavailable to CLI jobs unless explicitly configured.
        pass
    return None


def _alpaca_credentials() -> tuple[str, str]:
    pair = _streamlit_credential_pair()
    if pair:
        return pair

    # Non-Streamlit jobs can configure the standard environment variable pair.
    key = _clean_secret(os.getenv("APCA_API_KEY_ID"))
    secret = _clean_secret(os.getenv("APCA_API_SECRET_KEY"))
    if key and secret:
        return key, secret
    raise DataValidationError(
        "Could not read both Alpaca credentials from environment variables or Streamlit secrets."
    )


def _request_bars(params: dict[str, str]) -> tuple[list[dict[str, Any]], bytes, str]:
    key, secret = _alpaca_credentials()
    query = urllib.parse.urlencode(params)
    url = f"{ALPACA_DATA_URL}?{query}"
    request = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
        "User-Agent": "UNG-Trade-Desk/0.1",
    })
    all_bars: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    while True:
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
        except HTTPError as exc:
            if exc.code == 401:
                raise DataValidationError(
                    "Alpaca rejected the API credentials (HTTP 401). Check that APCA_API_KEY_ID "
                    "and APCA_API_SECRET_KEY are both correct and belong to the same Alpaca account. "
                    "Do not use placeholder text or include extra quote characters inside the values."
                ) from exc
            if exc.code == 403:
                raise DataValidationError(
                    "Alpaca denied this market-data request (HTTP 403). Check the account's data "
                    "feed entitlement and the requested SIP history window."
                ) from exc
            raise DataValidationError(f"Alpaca market-data request failed: HTTP {exc.code}") from exc
        except Exception as exc:
            raise DataValidationError(f"Alpaca market-data request failed: {exc}") from exc
        try:
            payload = json.loads(raw)
            bars = payload["bars"]
            if not isinstance(bars, list):
                raise TypeError("bars is not a list")
        except (KeyError, TypeError, ValueError) as exc:
            message = ""
            try:
                message = str(payload.get("message") or payload.get("error") or "")
            except UnboundLocalError:
                pass
            raise DataValidationError(f"Alpaca response schema invalid: {message or exc}") from exc
        payloads.append(raw)
        all_bars.extend(bars)
        token = payload.get("next_page_token")
        if not token:
            break
        params["page_token"] = str(token)
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(f"{ALPACA_DATA_URL}?{query}", headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
            "User-Agent": "UNG-Trade-Desk/0.1",
        })
    raw_all = b"\n".join(payloads)
    return all_bars, raw_all, url


def _bars_frame(bars: list[dict[str, Any]], *, intraday: bool = False) -> pd.DataFrame:
    if not bars:
        raise DataValidationError("Alpaca returned no UNG bars")
    try:
        frame = pd.DataFrame({
            "open": [bar["o"] for bar in bars],
            "high": [bar["h"] for bar in bars],
            "low": [bar["l"] for bar in bars],
            "close": [bar["c"] for bar in bars],
            "volume": [bar["v"] for bar in bars],
        }, index=pd.to_datetime([bar["t"] for bar in bars], utc=True))
        frame.index.name = "timestamp"
        frame = frame.sort_index()
        if frame.index.has_duplicates:
            raise ValueError("duplicate bar timestamps")
        frame = frame.apply(pd.to_numeric, errors="raise")
        if intraday:
            if (frame[["open", "high", "low", "close"]] <= 0).any().any():
                raise ValueError("intraday prices must be positive")
            if (frame["volume"] < 0).any():
                raise ValueError("intraday volume must be non-negative")
            if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
                raise ValueError("intraday high below open/close")
            if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
                raise ValueError("intraday low above open/close")
            return frame
        return validate_ohlcv(frame)
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"Alpaca OHLCV bars invalid: {exc}") from exc


def _snapshot(instrument: str, url: str, raw: bytes, published: date) -> SourceSnapshot:
    return SourceSnapshot(
        provider="alpaca",
        instrument=instrument,
        retrieved_at=datetime.now(timezone.utc),
        published_through=published,
        content_sha256=hashlib.sha256(raw).hexdigest(),
        source_url=url,
    )


def fetch_alpaca_ung_daily() -> FetchedFrame:
    """Fetch consolidated completed daily UNG bars using the Alpaca SIP feed.

    The free account can retrieve historical SIP observations once they are more
    than 15 minutes old. The current incomplete regular session is discarded.
    """
    now = datetime.now(timezone.utc)
    safe_end = now - timedelta(minutes=16)
    bars, raw, url = _request_bars({
        "timeframe": "1Day",
        "start": "2016-01-01T00:00:00Z",
        "end": safe_end.isoformat().replace("+00:00", "Z"),
        "feed": "sip",
        "sort": "asc",
        "limit": "10000",
    })
    frame = _bars_frame(bars)
    today_et = datetime.now(ET).date()
    frame = frame.loc[frame.index.date < today_et]
    if len(frame) < 500:
        raise DataValidationError(f"Alpaca returned too little UNG history: {len(frame)} rows")
    latest = frame.index[-1].date()
    if (today_et - latest).days > 5:
        raise DataValidationError(f"Alpaca UNG daily history is stale: {latest}")
    return FetchedFrame(frame, _snapshot("UNG", url, raw, latest), raw)


def fetch_alpaca_ung_intraday(*, days: int = 7) -> FetchedFrame:
    """Fetch 1-minute consolidated UNG bars through the delayed snapshot cutoff.

    Schedule this collector at 07:00 ET and 08:45 ET to obtain complete 06:45
    and 08:30 ET snapshots respectively on the free 15-minute-delayed SIP feed.
    """
    now_et = datetime.now(ET)
    if time(7, 0) <= now_et.time() <= time(7, 25):
        snapshot_slot, cutoff_time = "mid", time(6, 45)
    elif time(8, 45) <= now_et.time() <= time(9, 5):
        snapshot_slot, cutoff_time = "one_hour", time(8, 30)
    else:
        raise DataValidationError("outside-RTH fetch runs at 07:00 or 08:45 ET for the 15m delay")
    cutoff = datetime.combine(now_et.date(), cutoff_time, tzinfo=ET)
    start = datetime.combine(
        now_et.date() - timedelta(days=days), time.min, tzinfo=ET
    ).isoformat()
    end = cutoff.isoformat()
    bars, raw, url = _request_bars({
        "timeframe": "1Min",
        "start": start,
        "end": end,
        "feed": "sip",
        "sort": "asc",
        "limit": "10000",
    })
    frame = _bars_frame(bars, intraday=True)
    local_index = frame.index.tz_convert(ET)
    current_pre = frame.loc[
        (local_index.date == now_et.date())
        & (local_index.time >= time(4, 0))
        & (local_index < pd.Timestamp(cutoff))
    ]
    if current_pre.empty:
        raise DataValidationError(f"Alpaca returned no UNG premarket bars before {cutoff_time} ET")
    latest_bar_end = current_pre.index[-1] + pd.Timedelta(minutes=1)
    if pd.Timestamp(cutoff).tz_convert("UTC") - latest_bar_end > pd.Timedelta(minutes=15):
        raise DataValidationError("Alpaca latest premarket bar is more than 15 minutes stale")
    snapshot = _snapshot(f"UNG_1m_{snapshot_slot}", url, raw, now_et.date())
    return FetchedFrame(frame, snapshot, raw)


def compare_sources(first: FetchedFrame, second: FetchedFrame, *, tolerance: float = 0.03) -> str:
    """Compare two independent daily OHLCV histories when a second source is configured."""
    if first.snapshot.provider == second.snapshot.provider:
        raise DataValidationError("providers are not independent")
    if first.frame.index[-1] != second.frame.index[-1]:
        raise DataValidationError("providers disagree on latest session")
    common = first.frame[["close"]].join(second.frame[["close"]], lsuffix="_a", rsuffix="_b", how="inner")
    if len(common) < 500:
        raise DataValidationError(f"only {len(common)} overlapping sessions")
    difference = (common.close_a - common.close_b).abs() / common[["close_a", "close_b"]].abs().max(axis=1)
    if float(difference.median()) > tolerance:
        raise DataValidationError(f"median close disagreement {float(difference.median()):.2%}")
    latest = abs(float(common.iloc[-1].close_a) - float(common.iloc[-1].close_b)) / max(
        float(common.iloc[-1].max()), 1e-12
    )
    if latest > tolerance:
        raise DataValidationError(f"latest close disagreement {latest:.2%}")
    return f"confirmed {first.frame.index[-1].date()} with {len(common)} overlapping sessions"
