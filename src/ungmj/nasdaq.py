"""Nasdaq public historical OHLCV adapter."""
from __future__ import annotations

import json
from datetime import datetime, timezone
import urllib.parse

import pandas as pd

from .contracts import DataValidationError, validate_ohlcv
from .feeds import FetchedFrame, _get, _snapshot


def fetch_nasdaq_ung() -> FetchedFrame:
    """Fetch historical UNG OHLCV from Nasdaq's public historical endpoint."""
    today = datetime.now(timezone.utc).date()
    params = urllib.parse.urlencode({
        "assetclass": "etf",
        "fromdate": "2007-04-18",
        "todate": today.isoformat(),
        "limit": "5000",
    })
    url = f"https://api.nasdaq.com/api/quote/UNG/historical?{params}"
    raw = _get(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    })
    try:
        payload = json.loads(raw)
        data = payload.get("data") or {}
        table = data.get("tradesTable") or {}
        rows = table.get("rows") or []
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("empty history")
        aliases = {
            "".join(ch for ch in str(column).lower() if ch.isalnum()): column
            for column in frame.columns
        }
        date_col = aliases.get("date")
        open_col = aliases.get("open")
        high_col = aliases.get("high")
        low_col = aliases.get("low")
        close_col = aliases.get("close") or aliases.get("closelast") or aliases.get("lastsaleprice")
        volume_col = aliases.get("volume")
        columns = [date_col, open_col, high_col, low_col, close_col, volume_col]
        if any(column is None for column in columns):
            raise ValueError(f"OHLCV fields not present: {list(frame.columns)}")
        selected = frame[columns].copy()
        selected.columns = ["session", "open", "high", "low", "close", "volume"]
        selected["session"] = pd.to_datetime(selected["session"], errors="raise").dt.normalize()
        for column in ("open", "high", "low", "close", "volume"):
            selected[column] = pd.to_numeric(
                selected[column].astype(str).str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False).str.strip(),
                errors="raise",
            )
        selected = selected.set_index("session").sort_index()
        checked = validate_ohlcv(selected, instrument="UNG")
        # Never use a current-day bar that may still be forming during market hours.
        checked = checked.loc[checked.index.date < today]
        if checked.empty:
            raise ValueError("no completed daily sessions")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DataValidationError(f"Nasdaq history response schema/coverage invalid: {exc}") from exc
    if len(checked) < 500:
        raise DataValidationError(f"Nasdaq history unexpectedly short: {len(checked)} rows")
    age_days = (today - checked.index[-1].date()).days
    if age_days < 0 or age_days > 5:
        raise DataValidationError(f"Nasdaq UNG history is stale: {checked.index[-1].date()} ({age_days} calendar days old)")
    return FetchedFrame(
        checked, _snapshot("nasdaq", "UNG", url, raw, checked.index[-1].date()), raw
    )
