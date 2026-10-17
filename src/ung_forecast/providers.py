"""Initial public-data adapters for UNG and Henry Hub."""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from .contracts import DataValidationError, FetchedFrame, SourceSnapshot, validate_ohlcv


def _get(url: str, *, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "UNG-Forecast/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise DataValidationError(f"HTTP {response.status}: {url}")
            body = response.read()
    except Exception as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError(f"download failed: {url}: {exc}") from exc
    if not body:
        raise DataValidationError(f"empty response: {url}")
    return body


def _snapshot(provider: str, instrument: str, url: str, raw: bytes, published: date) -> SourceSnapshot:
    return SourceSnapshot(
        provider=provider,
        instrument=instrument,
        retrieved_at=datetime.now(timezone.utc),
        published_through=published,
        content_sha256=hashlib.sha256(raw).hexdigest(),
        source_url=url,
    )


def fetch_yahoo_ung() -> FetchedFrame:
    """Fetch completed daily UNG bars using explicit timestamps."""
    start = int(datetime(2007, 4, 18, tzinfo=timezone.utc).timestamp())
    end = int(datetime.now(timezone.utc).timestamp())
    query = urllib.parse.urlencode({"period1": start, "period2": end, "interval": "1d", "events": "div,splits"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/UNG?{query}"
    raw = _get(url)
    try:
        payload: dict[str, Any] = json.loads(raw)
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        frame = pd.DataFrame(
            {key: quote[key] for key in ("open", "high", "low", "close", "volume")},
            index=pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None),
        ).dropna()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DataValidationError("Yahoo response schema/coverage invalid") from exc
    frame = validate_ohlcv(frame)
    today = datetime.now(timezone.utc).date()
    frame = frame.loc[frame.index.date < today]
    if len(frame) < 500:
        raise DataValidationError(f"Yahoo returned too little history: {len(frame)} rows")
    latest = frame.index[-1].date()
    if (today - latest).days > 5:
        raise DataValidationError(f"Yahoo history is stale: {latest}")
    return FetchedFrame(frame, _snapshot("yahoo", "UNG", url, raw, latest), raw)


def fetch_nasdaq_ung() -> FetchedFrame:
    """Fetch completed daily UNG bars from Nasdaq's public historical endpoint."""
    today = datetime.now(timezone.utc).date()
    query = urllib.parse.urlencode({
        "assetclass": "etf", "fromdate": "2007-04-18", "todate": today.isoformat(), "limit": "5000"
    })
    url = f"https://api.nasdaq.com/api/quote/UNG/historical?{query}"
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        payload = json.loads(raw)
        rows = payload["data"]["tradesTable"]["rows"]
        frame = pd.DataFrame(rows)
        aliases = {"".join(c for c in str(x).lower() if c.isalnum()): x for x in frame.columns}
        names = {
            "session": aliases.get("date"), "open": aliases.get("open"),
            "high": aliases.get("high"), "low": aliases.get("low"),
            "close": aliases.get("close") or aliases.get("closelast") or aliases.get("lastsaleprice"),
            "volume": aliases.get("volume"),
        }
        if any(value is None for value in names.values()):
            raise ValueError(f"missing Nasdaq columns: {list(frame.columns)}")
        frame = frame[[names[key] for key in names]]
        frame.columns = list(names)
        frame["session"] = pd.to_datetime(frame["session"], errors="raise").dt.normalize()
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="raise")
        frame = frame.set_index("session").sort_index()
    except Exception as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError(f"Nasdaq response schema invalid: {exc}") from exc
    frame = validate_ohlcv(frame)
    frame = frame.loc[frame.index.date < today]
    if len(frame) < 500:
        raise DataValidationError(f"Nasdaq returned too little history: {len(frame)} rows")
    latest = frame.index[-1].date()
    if (today - latest).days > 5:
        raise DataValidationError(f"Nasdaq history is stale: {latest}")
    return FetchedFrame(frame, _snapshot("nasdaq", "UNG", url, raw, latest), raw)


def compare_sources(first: FetchedFrame, second: FetchedFrame, *, tolerance: float = 0.03) -> str:
    """Require fresh, independent, materially consistent source histories."""
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
    latest = abs(float(common.iloc[-1].close_a) - float(common.iloc[-1].close_b)) / max(float(common.iloc[-1].max()), 1e-12)
    if latest > tolerance:
        raise DataValidationError(f"latest close disagreement {latest:.2%}")
    return f"confirmed {first.frame.index[-1].date()} with {len(common)} overlapping sessions"
