"""Free-source market data ingestion. Providers remain unapproved until live checks pass."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from io import BytesIO, StringIO
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
    """Fetch official daily Henry Hub spot history; use free EIA API key or public EIA XLS."""
    key = api_key or os.getenv("EIA_API_KEY")
    if key:
        params = [
            ("api_key", key), ("frequency", "daily"), ("data[0]", "value"),
            ("facets[process][]", "RNGWHHD"), ("start", "1997-01-01"),
            ("sort[0][column]", "period"), ("sort[0][direction]", "asc"),
            ("offset", "0"), ("length", "50000"),
        ]
        url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/?" + urllib.parse.urlencode(params)
        raw = _get(url)
        try:
            payload = json.loads(raw)
            frame = pd.DataFrame(payload["response"]["data"])
            if not {"period", "value"}.issubset(frame.columns) or frame.empty:
                raise ValueError("missing period/value rows")
            frame = frame.rename(columns={"period": "session", "value": "henry_hub_spot"})
            frame["session"] = pd.to_datetime(frame["session"], errors="raise").dt.normalize()
            frame["henry_hub_spot"] = pd.to_numeric(frame["henry_hub_spot"], errors="raise")
            frame = frame.set_index("session").sort_index()
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError("EIA API response did not match expected Henry Hub daily schema") from exc
        provider = "eia_api"
        source_url = url.split("?")[0]
        raw_for_hash = raw.encode("utf-8")
    else:
        # Official EIA historical workbook is public and requires no account or API key.
        source_url = "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls"
        request = urllib.request.Request(
            source_url, headers={"User-Agent": "UNGMJ-research/0.1 (personal non-commercial use)"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw_for_hash = response.read()
            if not raw_for_hash:
                raise DataValidationError("EIA historical workbook is empty")
            workbook = pd.read_excel(BytesIO(raw_for_hash), sheet_name=0, header=None, engine="xlrd")
        except Exception as exc:
            raise DataValidationError(f"Could not download/parse official EIA workbook: {exc}") from exc
        header_row = None
        date_col = value_col = None
        for row_index, row in workbook.iterrows():
            values = [str(value).strip().lower() for value in row.tolist()]
            for col_index, value in enumerate(values):
                if value == "date":
                    date_col = col_index
                if "henry hub" in value and ("spot" in value or "price" in value):
                    value_col = col_index
            if date_col is not None and value_col is not None:
                header_row = row_index
                break
        if header_row is None:
            raise DataValidationError("EIA workbook header not recognized; parser fails closed")
        body = workbook.iloc[header_row + 1:, [date_col, value_col]].copy()
        body.columns = ["session", "henry_hub_spot"]
        body["session"] = pd.to_datetime(body["session"], errors="coerce").dt.normalize()
        body["henry_hub_spot"] = pd.to_numeric(body["henry_hub_spot"], errors="coerce")
        frame = body.dropna(subset=["session", "henry_hub_spot"]).set_index("session").sort_index()
        raw = "[binary EIA XLS workbook; SHA-256 recorded from exact downloaded bytes]"
        provider = "eia_bulk_xls"
    if frame.empty or frame.index.has_duplicates or frame["henry_hub_spot"].isna().any():
        raise DataValidationError("EIA Henry Hub data is empty, duplicated, or incomplete")
    if (frame["henry_hub_spot"] <= 0).any():
        raise DataValidationError("EIA Henry Hub spot prices must be positive")
    if len(frame) < 1000:
        raise DataValidationError(f"EIA Henry Hub history unexpectedly short: {len(frame)} rows")
    published = frame.index[-1].date()
    snapshot = SourceSnapshot(
        provider=provider, instrument="HENRY_HUB_SPOT",
        retrieved_at=datetime.now(timezone.utc), published_through=published,
        content_sha256=hashlib.sha256(raw_for_hash).hexdigest(), source_url=source_url,
    )
    return frame[["henry_hub_spot"]], snapshot, raw

