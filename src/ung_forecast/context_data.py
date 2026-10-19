"""Release-aware EIA storage and CFTC positioning ingestion."""
from __future__ import annotations

import json
import os
import io
import zipfile
from io import StringIO
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from typing import Any

import pandas as pd
from io import BytesIO

from .contracts import DataValidationError


@dataclass(frozen=True, slots=True)
class HenryHubSpotSnapshot:
    """Latest EIA Henry Hub spot observation for display and provenance."""

    observation_date: date
    spot_usd_per_mmbtu: float
    published_date: date
    retrieved_at: datetime
    rows: int
    source_url: str
    content_sha256: str


def _download_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "UNG-Forecast/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise DataValidationError(f"HTTP {response.status}: {url}")
            return json.loads(response.read())
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError(f"context download failed: {exc}") from exc


def fetch_eia_storage(api_key: str | None = None) -> pd.DataFrame:
    """Fetch EIA weekly Lower-48 working gas storage with release timestamps.

    The API key is intentionally required for this source. The returned index is
    the observation week; ``available_at`` is the earliest valid model timestamp.
    """
    key = api_key or os.getenv("EIA_API_KEY")
    if not key:
        return fetch_eia_storage_xls()
    params = [
        ("api_key", key), ("frequency", "weekly"), ("data[0]", "value"),
        ("facets[seriesId][]", "NW2_EPG0_SWO_R48_BCF"),
        ("start", "1993-01-01"), ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"), ("length", "5000"),
    ]
    url = "https://api.eia.gov/v2/natural-gas/stor/wngsr/data/?" + urllib.parse.urlencode(params)
    payload = _download_json(url)
    try:
        rows = payload["response"]["data"]
        frame = pd.DataFrame(rows)
        if not {"period", "value"}.issubset(frame.columns):
            raise ValueError("missing period/value")
        frame["observation_date"] = pd.to_datetime(frame["period"], errors="raise").dt.normalize()
        frame["storage_bcf"] = pd.to_numeric(frame["value"], errors="raise")
        # EIA's weekly report is released Thursday 10:30 ET. Use UTC explicitly.
        frame["available_at"] = frame["observation_date"] + pd.Timedelta(days=6, hours=15, minutes=30)
        frame = frame.set_index("observation_date").sort_index()
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"EIA storage schema invalid: {exc}") from exc
    if len(frame) < 100 or (frame["storage_bcf"] <= 0).any():
        raise DataValidationError("EIA storage history is empty, short, or non-positive")
    return frame[["storage_bcf", "available_at"]]


def fetch_eia_storage_xls() -> pd.DataFrame:
    """Fetch the official public EIA weekly-storage workbook without an API key."""
    url = "https://www.eia.gov/dnav/ng/xls/NG_STOR_WKLY_S1_W.xls"
    request = urllib.request.Request(url, headers={"User-Agent": "UNG-Forecast/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        workbook = pd.ExcelFile(BytesIO(raw), engine="xlrd")
        contents = pd.read_excel(workbook, sheet_name="Contents", header=None)
        data = pd.read_excel(workbook, sheet_name="Data 1", header=None)
        release_row = contents[contents.astype(str).apply(lambda row: row.str.contains("Release Date:").any(), axis=1)]
        if release_row.empty:
            raise ValueError("release date missing")
        release_text = release_row.iloc[0].dropna().astype(str).iloc[-1]
        release_date = pd.to_datetime(release_text, errors="raise")
        # Data 1 row 2 contains the column labels; row 3 onward contains values.
        if data.shape[1] < 2 or str(data.iloc[2, 0]).strip().lower() != "date":
            raise ValueError("unexpected Data 1 layout")
        frame = pd.DataFrame({
            "observation_date": pd.to_datetime(data.iloc[3:, 0], errors="coerce"),
            "storage_bcf": pd.to_numeric(data.iloc[3:, 1], errors="coerce"),
        }).dropna()
        frame["available_at"] = release_date.normalize() + pd.Timedelta(hours=15, minutes=30)
        frame = frame.set_index("observation_date").sort_index()
    except Exception as exc:
        raise DataValidationError(f"EIA storage workbook invalid: {exc}") from exc
    if len(frame) < 100 or (frame["storage_bcf"] <= 0).any():
        raise DataValidationError("EIA storage workbook is empty, short, or non-positive")
    return frame[["storage_bcf", "available_at"]]


def fetch_eia_henry_hub_spot() -> HenryHubSpotSnapshot:
    """Fetch the latest official daily Henry Hub spot value for dashboard display.

    The workbook provides a release date but not historical per-observation
    publication timestamps. This snapshot is therefore not passed into model
    features; the UI labels it as reference data only.
    """
    url = "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls"
    request = urllib.request.Request(url, headers={"User-Agent": "UNG-Forecast/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise DataValidationError(f"HTTP {response.status}: EIA Henry Hub workbook")
            raw = response.read()
        workbook = pd.ExcelFile(BytesIO(raw), engine="xlrd")
        contents = pd.read_excel(workbook, sheet_name="Contents", header=None)
        data = pd.read_excel(workbook, sheet_name="Data 1", header=None)
        release_rows = contents[
            contents.astype(str).apply(
                lambda row: row.str.contains("Release Date:", regex=False).any(), axis=1
            )
        ]
        if release_rows.empty:
            raise ValueError("release date missing")
        release_text = release_rows.iloc[0].dropna().astype(str).iloc[-1]
        published_date = pd.to_datetime(release_text, errors="raise").date()
        if data.shape[1] < 2 or str(data.iloc[2, 0]).strip().lower() != "date":
            raise ValueError("unexpected Data 1 layout")
        frame = pd.DataFrame({
            "observation_date": pd.to_datetime(data.iloc[3:, 0], errors="coerce"),
            "spot_usd_per_mmbtu": pd.to_numeric(data.iloc[3:, 1], errors="coerce"),
        }).dropna()
        frame = frame.sort_values("observation_date")
        if frame.empty or frame["observation_date"].duplicated().any():
            raise ValueError("Henry Hub history is empty or duplicated")
        latest = frame.iloc[-1]
        value = float(latest["spot_usd_per_mmbtu"])
        observation_date = pd.Timestamp(latest["observation_date"]).date()
        if value <= 0 or observation_date > published_date:
            raise ValueError("Henry Hub latest observation is invalid")
    except Exception as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError(f"EIA Henry Hub workbook invalid: {type(exc).__name__}: {exc}") from exc
    return HenryHubSpotSnapshot(
        observation_date=observation_date,
        spot_usd_per_mmbtu=value,
        published_date=published_date,
        retrieved_at=datetime.now(timezone.utc),
        rows=len(frame),
        source_url=url,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def make_point_in_time_context(
    context: pd.DataFrame, sessions: pd.DatetimeIndex, *, asof_column: str = "available_at"
) -> pd.DataFrame:
    """Carry released context values forward without using unreleased observations."""
    if asof_column not in context.columns:
        raise DataValidationError(f"context is missing {asof_column}")
    context = context.sort_values(asof_column).copy()
    left = pd.DataFrame({"session": sessions}).sort_values("session")
    right = context.reset_index(drop=False).rename(columns={context.index.name or "index": "observation_date"})
    right[asof_column] = pd.to_datetime(right[asof_column], utc=True).dt.tz_convert(None)
    left["session"] = pd.to_datetime(left["session"]).dt.tz_localize(None)
    return pd.merge_asof(left, right, left_on="session", right_on=asof_column, direction="backward").set_index("session")


def parse_cftc_positioning(csv_text: str, *, release_hour_utc: int = 21) -> pd.DataFrame:
    """Parse a normalized CFTC disaggregated natural-gas CSV export."""
    try:
        raw = pd.read_csv(StringIO(csv_text))
        aliases = {str(column).strip().lower().replace(" ", "_"): column for column in raw.columns}
        date_col = aliases.get("report_date") or aliases.get("date")
        oi_col = aliases.get("open_interest")
        managed_long = aliases.get("managed_money_long")
        managed_short = aliases.get("managed_money_short")
        producer_long = aliases.get("producer_merchant_long")
        producer_short = aliases.get("producer_merchant_short")
        required = [date_col, oi_col, managed_long, managed_short, producer_long, producer_short]
        if any(column is None for column in required):
            raise ValueError(f"required CFTC columns missing: {list(raw.columns)}")
        frame = pd.DataFrame({
            "report_date": pd.to_datetime(raw[date_col], errors="raise").dt.normalize(),
            "open_interest": pd.to_numeric(raw[oi_col], errors="raise"),
            "managed_money_net": pd.to_numeric(raw[managed_long], errors="raise") - pd.to_numeric(raw[managed_short], errors="raise"),
            "producer_merchant_net": pd.to_numeric(raw[producer_long], errors="raise") - pd.to_numeric(raw[producer_short], errors="raise"),
        })
        frame["available_at"] = frame["report_date"] + pd.Timedelta(days=4, hours=release_hour_utc)
        frame = frame.set_index("report_date").sort_index()
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"CFTC positioning schema invalid: {exc}") from exc
    if frame.empty or frame.index.has_duplicates or (frame["open_interest"] <= 0).any():
        raise DataValidationError("CFTC positioning is empty, duplicated, or invalid")
    frame["managed_money_net_pct_oi"] = frame["managed_money_net"] / frame["open_interest"]
    frame["producer_merchant_net_pct_oi"] = frame["producer_merchant_net"] / frame["open_interest"]
    return frame


def fetch_cftc_positioning(url: str | None = None) -> pd.DataFrame:
    """Fetch a configured official CFTC CSV export and parse it strictly."""
    source_url = url or os.getenv("CFTC_NATURAL_GAS_CSV_URL")
    if not source_url:
        return fetch_cftc_official_history()
    request = urllib.request.Request(source_url, headers={"User-Agent": "UNG-Forecast/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8-sig")
    except Exception as exc:
        raise DataValidationError(f"CFTC download failed: {exc}") from exc
    return parse_cftc_positioning(raw)


def fetch_cftc_official_history(years: int = 5) -> pd.DataFrame:
    """Fetch recent official CFTC annual archives for NYMEX natural gas."""
    if years < 1 or years > 20:
        raise ValueError("years must be between 1 and 20")
    current_year = datetime.now(timezone.utc).year
    frames = []
    for year in range(current_year - years + 1, current_year + 1):
        url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
        request = urllib.request.Request(url, headers={"User-Agent": "UNG-Forecast/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                member = next(name for name in archive.namelist() if name.lower().endswith("annual.txt"))
                frame = pd.read_csv(archive.open(member), low_memory=False)
        except (StopIteration, OSError, ValueError) as exc:
            raise DataValidationError(f"CFTC archive {year} invalid: {exc}") from exc
        name_column = frame.columns[0]
        selected = frame[frame[name_column].astype(str).str.strip().eq("NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE")]
        if not selected.empty:
            selected = selected.rename(columns={
                "As of Date in Form YYYY-MM-DD": "report_date",
                "Open Interest (All)": "open_interest",
                "Noncommercial Positions-Long (All)": "managed_money_long",
                "Noncommercial Positions-Short (All)": "managed_money_short",
                "Commercial Positions-Long (All)": "producer_merchant_long",
                "Commercial Positions-Short (All)": "producer_merchant_short",
            })
            frames.append(selected[["report_date", "open_interest", "managed_money_long", "managed_money_short", "producer_merchant_long", "producer_merchant_short"]])
    if not frames:
        raise DataValidationError("CFTC archives contained no NYMEX natural-gas rows")
    return parse_cftc_positioning(pd.concat(frames, ignore_index=True).to_csv(index=False))


def combine_context_sources(frames: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Combine release-aware context frames into one availability-gated frame."""
    if not frames:
        return None
    result = frames[0].copy()
    for frame in frames[1:]:
        result = result.join(frame, how="outer", lsuffix="", rsuffix="_other")
        if "available_at_other" in result.columns:
            result["available_at"] = result[["available_at", "available_at_other"]].max(axis=1)
            result = result.drop(columns=["available_at_other"])
    return result.sort_index()
