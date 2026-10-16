"""Free-source market data ingestion. Providers remain unapproved until live checks pass."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from io import StringIO
from typing import Any

import pandas as pd

from .contracts import DataValidationError, SourceSnapshot, validate_ohlcv


@dataclass(frozen=True, slots=True)
class FetchedFrame:
    frame: pd.DataFrame
    snapshot: SourceSnapshot
    raw_text: str


def _get(url: str, *, headers: dict[str, str] | None = None, timeout: int = 25) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "UNGMJ-research/0.1 (personal non-commercial use)", **(headers or {})}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise DataValidationError(f"HTTP {response.status} from {url}")
        body = response.read()
    if not body:
        raise DataValidationError(f"Empty response from {url}")
    return body.decode("utf-8-sig")


def _snapshot(provider: str, instrument: str, url: str, raw: str, published: date | None) -> SourceSnapshot:
    return SourceSnapshot(
        provider=provider,
        instrument=instrument,
        retrieved_at=datetime.now(timezone.utc),
        published_through=published,
        content_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        source_url=url,
    )


def fetch_stooq_ung() -> FetchedFrame:
    """Fetch daily UNG bars from Stooq's public CSV endpoint; availability is not guaranteed."""
    url = "https://stooq.com/q/d/l/?s=ung.us&i=d"
    raw = _get(url)
    if raw.strip().lower().startswith("no data"):
        raise DataValidationError("Stooq returned no UNG data")
    try:
        frame = pd.read_csv(StringIO(raw))
    except Exception as exc:
        raise DataValidationError("Could not parse Stooq CSV") from exc
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(frame.columns):
        raise DataValidationError(f"Stooq CSV schema mismatch: {list(frame.columns)}")
    frame = frame.rename(columns={
        "Date": "session", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume"
    }).set_index("session")
    checked = validate_ohlcv(frame, instrument="UNG")
    return FetchedFrame(checked, _snapshot("stooq", "UNG", url, raw, checked.index[-1].date()), raw)


def fetch_yahoo_ung() -> FetchedFrame:
    """Fetch unadjusted daily UNG OHLCV from Yahoo's public chart endpoint."""
    params = urllib.parse.urlencode({"range": "max", "interval": "1d", "events": "div,splits"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/UNG?{params}"
    raw = _get(url)
    try:
        payload: dict[str, Any] = json.loads(raw)
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        frame = pd.DataFrame({
            "open": quote["open"], "high": quote["high"], "low": quote["low"],
            "close": quote["close"], "volume": quote["volume"],
        }, index=pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DataValidationError("Yahoo chart response schema/coverage invalid") from exc
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    checked = validate_ohlcv(frame, instrument="UNG")
    return FetchedFrame(checked, _snapshot("yahoo_chart", "UNG", url, raw, checked.index[-1].date()), raw)


def fetch_eia_henry_hub_spot(api_key: str | None = None) -> tuple[pd.DataFrame, SourceSnapshot, str]:
    """Fetch EIA daily Henry Hub spot price through EIA v2; API key is free but required."""
    key = api_key or os.getenv("EIA_API_KEY")
    if not key:
        raise DataValidationError("EIA_API_KEY is not configured; Henry Hub spot is a required input")
    params = [
        ("api_key", key), ("frequency", "daily"), ("data[0]", "value"),
        ("facets[process][]", "RNGWHHD"), ("start", "1997-01-01"),
        ("sort[0][column]", "period"), ("sort[0][direction]", "asc"),
        ("offset", "0"), ("length", "5000"),
    ]
    url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/?" + urllib.parse.urlencode(params)
    raw = _get(url)
    try:
        payload = json.loads(raw)
        rows = payload["response"]["data"]
        frame = pd.DataFrame(rows)
        if not {"period", "value"}.issubset(frame.columns) or frame.empty:
            raise ValueError("missing period/value rows")
        frame = frame.rename(columns={"period": "session", "value": "henry_hub_spot"})
        frame["session"] = pd.to_datetime(frame["session"], errors="raise").dt.normalize()
        frame["henry_hub_spot"] = pd.to_numeric(frame["henry_hub_spot"], errors="raise")
        frame = frame.set_index("session").sort_index()
        if frame.index.has_duplicates or not frame["henry_hub_spot"].map(pd.notna).all():
            raise ValueError("duplicate or missing values")
        if (frame["henry_hub_spot"] <= 0).any():
            raise ValueError("non-positive spot prices")
        published = frame.index[-1].date()
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError("EIA response did not match the expected Henry Hub daily schema") from exc
    snapshot = _snapshot("eia", "HENRY_HUB_SPOT", url.split("?")[0], raw, published)
    return frame[["henry_hub_spot"]], snapshot, raw


def latest_session_agreement(
    first: FetchedFrame, second: FetchedFrame, *, close_tolerance: float = 0.03
) -> tuple[bool, str]:
    """Check same latest session and close agreement; tolerance is a relative fraction."""
    a, b = first.frame, second.frame
    if first.snapshot.provider == second.snapshot.provider:
        return False, "Providers are not independent"
    if a.index[-1] != b.index[-1]:
        return False, f"Latest sessions disagree: {a.index[-1].date()} vs {b.index[-1].date()}"
    left, right = float(a.iloc[-1]["close"]), float(b.iloc[-1]["close"])
    relative = abs(left - right) / max(abs(left), abs(right))
    if relative > close_tolerance:
        return False, f"Latest closes differ by {relative:.2%}, exceeding {close_tolerance:.2%}"
    return True, f"Confirmed session {a.index[-1].date()}; relative close difference {relative:.3%}"


def fetch_ung_sources() -> tuple[FetchedFrame, FetchedFrame]:
    """Fetch both candidate sources. A failure must be surfaced; never silently downgrade."""
    errors: list[str] = []
    results: list[FetchedFrame] = []
    for fetcher in (fetch_yahoo_ung, fetch_stooq_ung):
        try:
            results.append(fetcher())
        except Exception as exc:
            errors.append(f"{fetcher.__name__}: {type(exc).__name__}: {exc}")
    if len(results) != 2:
        raise DataValidationError("Both independent UNG sources are required for confirmation. " + " | ".join(errors))
    return results[0], results[1]
