"""Serializable feed summaries for the research dashboard and run report."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .context_data import HenryHubSpotSnapshot
from .contracts import FetchedFrame


def _utc_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def describe_price_feed(name: str, fetched: FetchedFrame, role: str) -> dict[str, Any]:
    frame = fetched.frame
    latest = frame.iloc[-1]
    return {
        "source": name,
        "status": "AVAILABLE",
        "latest_observation": str(frame.index[-1].date()),
        "latest_value": f"${float(latest['close']):.2f} close",
        "available_at": "Completed daily session; source does not expose a precise publication time",
        "retrieved_at": _utc_iso(fetched.snapshot.retrieved_at),
        "rows": len(frame),
        "role": role,
        "source_url": fetched.snapshot.source_url,
        "note": "",
    }


def describe_context_feed(
    name: str,
    frame: pd.DataFrame,
    *,
    value_columns: tuple[str, ...],
    availability_note: str,
    source_url: str,
) -> dict[str, Any]:
    latest = frame.iloc[-1]
    values = []
    for column in value_columns:
        if column not in frame.columns or pd.isna(latest[column]):
            continue
        value = float(latest[column])
        if column == "storage_bcf":
            values.append(f"{value:,.0f} Bcf")
        elif column.endswith("_pct_oi"):
            values.append(f"{column.replace('_pct_oi', '')}: {value:.1%} of OI")
        else:
            values.append(f"{column}: {value:.4g}")
    observation = pd.Timestamp(frame.index[-1]).date().isoformat()
    available_at = _utc_iso(latest.get("available_at"))
    return {
        "source": name,
        "status": "AVAILABLE",
        "latest_observation": observation,
        "latest_value": "; ".join(values) or "",
        "available_at": available_at or "Unavailable",
        "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows": len(frame),
        "role": "Context feed",
        "source_url": source_url,
        "note": availability_note,
    }


def describe_henry_hub_feed(snapshot: HenryHubSpotSnapshot) -> dict[str, Any]:
    return {
        "source": "EIA Henry Hub spot",
        "status": "AVAILABLE",
        "latest_observation": snapshot.observation_date.isoformat(),
        "latest_value": f"${snapshot.spot_usd_per_mmbtu:.2f}/MMBtu",
        "available_at": f"EIA workbook release date: {snapshot.published_date.isoformat()} (date only)",
        "retrieved_at": _utc_iso(snapshot.retrieved_at),
        "rows": snapshot.rows,
        "role": "Reference only; excluded from forecast features",
        "source_url": snapshot.source_url,
        "note": "Workbook lacks per-observation publication timestamps; historical point-in-time use is not validated.",
    }


def unavailable_feed(source: str, note: str, *, status: str = "UNAVAILABLE") -> dict[str, Any]:
    return {
        "source": source,
        "status": status,
        "latest_observation": "—",
        "latest_value": "—",
        "available_at": "—",
        "retrieved_at": None,
        "rows": 0,
        "role": "Not used by forecast",
        "source_url": "",
        "note": note,
    }
