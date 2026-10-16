"""Nasdaq public-history adapter for the UNG ETF."""
from __future__ import annotations

import json
import pandas as pd

from .contracts import DataValidationError, validate_ohlcv
from .feeds import FetchedFrame, _get, _snapshot


def fetch_nasdaq_ung() -> FetchedFrame:
    url = "https://api.nasdaq.com/api/quote/UNG/historical?assetclass=etf&limit=5000"
    raw = _get(url, headers={
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    })
    try:
        payload = json.loads(raw)
        rows = payload["data"]["tradesTable"]["rows"]
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("empty history")
        aliases = {
            "".join(ch for ch in str(col).lower() if ch.isalnum()): col
            for col in frame.columns
        }
        def col(*names: str) -> str:
            for name in names:
                if name in aliases:
                    return aliases[name]
            raise ValueError("required OHLCV column missing")
        selected = frame[[col("date"), col("open"), col("high"), col("low"),
                          col("close", "closelast", "lastsaleprice"), col("volume")]].copy()
        selected.columns = ["session", "open", "high", "low", "close", "volume"]
        selected["session"] = pd.to_datetime(selected["session"], errors="raise").dt.normalize()
        for field in ("open", "high", "low", "close", "volume"):
            selected[field] = pd.to_numeric(
                selected[field].astype(str).str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False).str.strip(), errors="raise"
            )
        checked = validate_ohlcv(selected.set_index("session"), instrument="UNG")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DataValidationError("Nasdaq history response schema/coverage invalid") from exc
    if len(checked) < 500:
        raise DataValidationError("Nasdaq history unexpectedly short")
    return FetchedFrame(
        checked, _snapshot("nasdaq", "UNG", url, raw, checked.index[-1].date()), raw
    )
