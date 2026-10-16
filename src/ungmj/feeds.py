"""Free-source market data ingestion. Providers remain unapproved until live checks pass."""
from __future__ import annotations

import hashlib
import logging
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

LOG = logging.getLogger(__name__)


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
    age_days = (datetime.now(timezone.utc).date() - checked.index[-1].date()).days
    if age_days < 0 or age_days > 5:
        raise DataValidationError(f"Yahoo UNG history is stale: {checked.index[-1].date()} ({age_days} calendar days old)")
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
        import re
        weekday_names = ("mon", "tue", "wed", "thu", "fri")
        header_row = None
        week_col = None
        weekday_cols: dict[str, int] = {}
        for row_index, row in workbook.iterrows():
            values = [str(value).replace("chr(160)", " ").strip().lower() for value in row.tolist()]
            if "week of" in values:
                header_row = row_index
                week_col = values.index("week of")
                for weekday in weekday_names:
                    if weekday in values:
                        weekday_cols[weekday] = values.index(weekday)
                if len(weekday_cols) == 5:
                    break
        if header_row is None or week_col is None or len(weekday_cols) != 5:
            sample = workbook.head(10).fillna("").astype(str).to_dict(orient="records")
            raise DataValidationError(
                f"EIA workbook weekly header not recognized; shape={workbook.shape}; sample={sample}"
            )
        month_numbers = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        pattern = re.compile(
            r"(?P<year>\d{4})\s+(?P<month>[A-Za-z]{3})-\s*(?P<day>\d{1,2})\s+to\s+"
            r"(?P<end_month>[A-Za-z]{3})-\s*(?P<end_day>\d{1,2})",
            flags=re.IGNORECASE,
        )
        observations: list[tuple[datetime, float]] = []
        for _, row in workbook.iloc[header_row + 1:].iterrows():
            match = pattern.search(str(row.iloc[week_col]).replace("chr(160)", " ").strip())
            if not match:
                continue
            year = int(match.group("year"))
            month = month_numbers.get(match.group("month").lower())
            if month is None:
                continue
            week_start = datetime(year, month, int(match.group("day")))
            for day_offset, weekday in enumerate(weekday_names):
                raw_value = str(row.iloc[weekday_cols[weekday]]).replace("$", "").replace(",", "").strip()
                value = pd.to_numeric(raw_value, errors="coerce")
                if pd.isna(value):
                    continue
                session = week_start + pd.Timedelta(days=day_offset)
                observations.append((pd.Timestamp(session).to_pydatetime(), float(value)))
        if not observations:
            raise DataValidationError("EIA workbook contains no parseable daily Henry Hub observations")
        frame = pd.DataFrame(observations, columns=["session", "henry_hub_spot"]).set_index("session").sort_index()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        if len(frame) < 1000:
            raise DataValidationError(f"EIA workbook parsed only {len(frame)} daily observations")
        raw = "[binary EIA XLS workbook; SHA-256 recorded from exact downloaded bytes]"
        provider = "eia_bulk_xls"
    if frame.empty or frame.index.has_duplicates or frame["henry_hub_spot"].isna().any():
        raise DataValidationError("EIA Henry Hub data is empty, duplicated, or incomplete")
    if (frame["henry_hub_spot"] <= 0).any():
        raise DataValidationError("EIA Henry Hub spot prices must be positive")
    if len(frame) < 1000:
        raise DataValidationError(f"EIA Henry Hub history unexpectedly short: {len(frame)} rows")
    published = frame.index[-1].date()
    age_days = (datetime.now(timezone.utc).date() - published).days
    if age_days < 0 or age_days > 7:
        raise DataValidationError(f"EIA Henry Hub spot data is stale: {published} ({age_days} calendar days old)")
    snapshot = SourceSnapshot(
        provider=provider, instrument="HENRY_HUB_SPOT",
        retrieved_at=datetime.now(timezone.utc), published_through=published,
        content_sha256=hashlib.sha256(raw_for_hash).hexdigest(), source_url=source_url,
    )
    return frame[["henry_hub_spot"]], snapshot, raw



def latest_session_agreement(
    first: FetchedFrame, second: FetchedFrame, *, close_tolerance: float = 0.03
) -> tuple[bool, str]:
    """Require fresh, overlapping, price-consistent histories from distinct distributors."""
    a, b = first.frame, second.frame
    if first.snapshot.provider == second.snapshot.provider:
        return False, "Providers are not independent"
    if a.empty or b.empty:
        return False, "One provider returned an empty dataset"
    if a.index[-1] != b.index[-1]:
        return False, f"Latest sessions disagree: {a.index[-1].date()} vs {b.index[-1].date()}"
    age_days = (datetime.now(timezone.utc).date() - a.index[-1].date()).days
    if age_days < 0 or age_days > 5:
        return False, f"Latest UNG session is stale: {a.index[-1].date()} ({age_days} calendar days old)"
    common = a[["close"]].join(b[["close"]], how="inner", lsuffix="_a", rsuffix="_b")
    if len(common) < 500:
        return False, f"Insufficient historical overlap: {len(common)} sessions; need 500"
    history_gap = (
        (common["close_a"] - common["close_b"]).abs()
        / common[["close_a", "close_b"]].abs().max(axis=1)
    ).median()
    if float(history_gap) > close_tolerance:
        return False, (
            f"Historical close disagreement median {history_gap:.2%} "
            f"exceeds {close_tolerance:.2%}"
        )
    left, right = float(a.iloc[-1]["close"]), float(b.iloc[-1]["close"])
    relative = abs(left - right) / max(abs(left), abs(right))
    if relative > close_tolerance:
        return False, f"Latest closes differ by {relative:.2%}, exceeding {close_tolerance:.2%}"
    return True, (
        f"Confirmed session {a.index[-1].date()}; latest close difference {relative:.3%}; "
        f"median historical difference {history_gap:.3%}"
    )


def fetch_ung_sources() -> tuple[FetchedFrame, FetchedFrame | None]:
    """Fetch a primary UNG OHLCV source and optional independent confirmation source."""
    try:
        primary = fetch_yahoo_ung()
    except Exception as exc:
        raise DataValidationError(
            f"Primary Yahoo UNG source failed: {type(exc).__name__}: {exc}"
        ) from exc
    from .nasdaq import fetch_nasdaq_ung
    errors: list[str] = []
    for fetcher in (fetch_nasdaq_ung, fetch_stooq_ung):
        try:
            return primary, fetcher()
        except Exception as exc:
            errors.append(f"{fetcher.__name__}: {type(exc).__name__}: {exc}")
    LOG.warning("Secondary UNG providers unavailable; primary-only forecasts must remain PROVISIONAL: %s", " | ".join(errors))
    return primary, None
