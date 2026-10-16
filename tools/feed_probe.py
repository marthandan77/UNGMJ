#!/usr/bin/env python3
"""Manual, read-only public-data feed probe for UNG research.

This script fetches a small set of candidate public endpoints only when the
GitHub Actions workflow is manually run. It does not trade, place orders, or
write forecasts. It writes a machine-readable report to artifacts/feed-probe.json.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

OUT = Path("artifacts/feed-probe.json")
UA = "UNGMJ-public-feed-probe/0.1 (research; read-only)"
TIMEOUT_SECONDS = 25
MIN_UNG_ROWS = 500


def fetch(url: str) -> dict:
    started = time.monotonic()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Accept": "*/*", "Cache-Control": "no-cache"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(20_000_000)
            return {
                "ok": 200 <= response.status < 300,
                "http_status": response.status,
                "final_url": response.geturl(),
                "content_type": response.headers.get("Content-Type", ""),
                "last_modified": response.headers.get("Last-Modified"),
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "body": body,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(200_000)
        return {
            "ok": False, "http_status": exc.code, "final_url": exc.geturl(),
            "content_type": exc.headers.get("Content-Type", ""),
            "last_modified": exc.headers.get("Last-Modified"),
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "body": body, "error": f"HTTPError: {exc.code}",
        }
    except Exception as exc:  # Network/provider errors are evidence, not crashes.
        return {
            "ok": False, "http_status": None, "final_url": url,
            "content_type": "", "last_modified": None, "bytes": 0,
            "sha256": None, "elapsed_ms": round((time.monotonic() - started) * 1000),
            "body": b"", "error": f"{type(exc).__name__}: {exc}",
        }


def public_result(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if k != "body"}


def probe_yahoo_ung() -> dict:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/UNG?range=5y&interval=1d&events=div%2Csplits"
    raw = fetch(url)
    result = {"provider": "Yahoo Finance chart endpoint (unofficial)", "url": url,
              "status": "FAIL", "checks": [], "transport": public_result(raw)}
    if not raw["ok"]:
        result["checks"].append({"name": "http_success", "pass": False, "detail": raw["error"] or raw["http_status"]})
        return result
    try:
        payload = json.loads(raw["body"])
        chart = payload.get("chart", {})
        if chart.get("error"):
            raise ValueError(f"provider chart error: {chart['error']}")
        item = (chart.get("result") or [None])[0]
        if not item:
            raise ValueError("no chart result returned")
        timestamps = item.get("timestamp") or []
        quote = ((item.get("indicators") or {}).get("quote") or [{}])[0]
        opens, highs, lows, closes, volumes = [
            quote.get(k) or [] for k in ("open", "high", "low", "close", "volume")
        ]
        rows = []
        for i, ts in enumerate(timestamps):
            vals = [a[i] if i < len(a) else None for a in (opens, highs, lows, closes, volumes)]
            if any(v is None for v in vals[:4]):
                continue
            dt = datetime.fromtimestamp(ts, timezone.utc).date().isoformat()
            o, h, l, c = map(float, vals[:4])
            vol = vals[4]
            rows.append({"date": dt, "open": o, "high": h, "low": l, "close": c,
                         "volume": None if vol is None else int(vol)})
        dates = [r["date"] for r in rows]
        unique_dates = len(dates) == len(set(dates))
        ordered = dates == sorted(dates)
        positive = all(min(r["open"], r["high"], r["low"], r["close"]) > 0 for r in rows)
        ohlc = all(r["high"] >= max(r["open"], r["close"], r["low"]) and
                   r["low"] <= min(r["open"], r["close"], r["high"]) for r in rows)
        volume_ok = all(r["volume"] is None or r["volume"] >= 0 for r in rows)
        latest = max(dates) if dates else None
        latest_age_days = None
        if latest:
            latest_age_days = (datetime.now(timezone.utc).date() -
                               datetime.fromisoformat(latest).date()).days
        checks = [
            ("chart_payload_present", bool(item), "A chart result was returned."),
            ("minimum_history", len(rows) >= MIN_UNG_ROWS, f"{len(rows)} valid OHLC rows; minimum {MIN_UNG_ROWS}."),
            ("unique_dates", unique_dates, "No duplicate daily dates." if unique_dates else "Duplicate dates detected."),
            ("chronological_order", ordered, "Dates are chronological." if ordered else "Dates are not chronological."),
            ("positive_prices", positive, "All OHLC prices are positive." if positive else "Non-positive OHLC value detected."),
            ("ohlc_consistency", ohlc, "High/low relationships are valid." if ohlc else "Impossible OHLC relationship detected."),
            ("volume_nonnegative", volume_ok, "Volume is nonnegative or unavailable." if volume_ok else "Negative volume detected."),
            ("freshness_7_calendar_days", latest_age_days is not None and 0 <= latest_age_days <= 7,
             f"Latest date {latest}; age {latest_age_days} calendar days (weekends/holidays are not specially modelled)."),
        ]
        result.update({
            "checks": [{"name": n, "pass": bool(p), "detail": d} for n, p, d in checks],
            "row_count": len(rows), "first_date": min(dates) if dates else None,
            "latest_date": latest, "latest_close": rows[-1]["close"] if rows else None,
            "latest_volume": rows[-1]["volume"] if rows else None,
            "sample_tail": rows[-3:],
            "status": "PASS" if all(c[1] for c in checks) else "FAIL",
            "data_note": "UNG is the target ETF. This endpoint is unofficial and can change without notice."
        })
    except Exception as exc:
        result["checks"].append({"name": "parse_payload", "pass": False,
                                 "detail": f"{type(exc).__name__}: {exc}"})
    return result


def probe_stooq_ung() -> dict:
    url = "https://stooq.com/q/d/l/?s=ung.us&i=d"
    raw = fetch(url)
    result = {"provider": "Stooq CSV (access not guaranteed)", "url": url,
              "status": "FAIL", "transport": public_result(raw), "checks": []}
    if not raw["ok"]:
        result["checks"].append({"name": "http_success", "pass": False, "detail": raw["error"] or raw["http_status"]})
        return result
    text_body = raw["body"].decode("utf-8-sig", errors="replace")
    try:
        rows = list(csv.DictReader(io.StringIO(text_body)))
        header = set(rows[0].keys()) if rows else set()
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        header_ok = required.issubset(header)
        dates = [r.get("Date", "") for r in rows]
        parse_ok = bool(rows) and header_ok and all(r.get("Close") for r in rows)
        checks = [
            {"name": "csv_header", "pass": header_ok, "detail": f"Columns found: {sorted(header)}"},
            {"name": "data_rows", "pass": len(rows) >= MIN_UNG_ROWS, "detail": f"{len(rows)} rows; minimum {MIN_UNG_ROWS}."},
            {"name": "date_order_and_uniqueness", "pass": len(dates) == len(set(dates)) and dates == sorted(dates),
             "detail": "Dates unique and ordered." if len(dates) == len(set(dates)) and dates == sorted(dates) else "Dates are duplicated or unordered."},
        ]
        result.update({"checks": checks, "row_count": len(rows),
                       "first_date": dates[0] if dates else None,
                       "latest_date": dates[-1] if dates else None,
                       "sample_tail": rows[-3:], "status": "PASS" if parse_ok and all(c["pass"] for c in checks) else "FAIL"})
        if not rows:
            result["preview"] = text_body[:300]
    except Exception as exc:
        result["checks"].append({"name": "parse_csv", "pass": False, "detail": f"{type(exc).__name__}: {exc}"})
    return result


def probe_eia_henry_hub_bulk() -> dict:
    # EIA's downloadable daily historical workbook; no API key is used here.
    url = "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls"
    raw = fetch(url)
    body = raw["body"]
    xls_signature = body.startswith(bytes.fromhex("D0CF11E0A1B11AE1"))
    zip_signature = body.startswith(b"PK\x03\x04")  # tolerate an XLSX redirect/change
    html_error = b"<html" in body[:500].lower() or b"<!doctype html" in body[:500].lower()
    checks = [
        {"name": "http_success", "pass": raw["ok"], "detail": raw["error"] or str(raw["http_status"])},
        {"name": "workbook_signature", "pass": xls_signature or zip_signature,
         "detail": "Excel workbook signature found." if xls_signature or zip_signature else
                   ("HTML response; likely block/error page." if html_error else "Response is not a recognized Excel workbook.")},
        {"name": "nontrivial_size", "pass": len(body) > 10_000, "detail": f"{len(body)} bytes received."},
    ]
    return {"provider": "U.S. EIA Henry Hub daily historical workbook", "url": url,
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
            "transport": public_result(raw), "checks": checks,
            "data_note": "This probe checks workbook availability/signature only; production parsing and date/value validation are still required."}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "project": "UNGMJ",
        "probe_type": "manual_public_feed_scan",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Read-only endpoint and data-shape checks; not a forecast and not trading advice.",
        "results": [probe_yahoo_ung(), probe_stooq_ung(), probe_eia_henry_hub_bulk()],
        "selection_rule": "No provider is promoted automatically. Review results; a passing endpoint is a candidate, not yet production-approved.",
    }
    for item in report["results"]:
        item["transport"].pop("body", None)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for item in report["results"]:
        print(f'{item["status"]:4} | {item["provider"]}')
        for check in item.get("checks", []):
            mark = "PASS" if check.get("pass") else "FAIL"
            print(f'  {mark}: {check["name"]} — {check.get("detail", "")}')
    print(f"Report written to {OUT}")
    # Return success so all providers are reported even when a candidate is down.
    # Provider status is recorded in the JSON artifact, not hidden by early exit.
    return 0


if __name__ == "__main__":
    sys.exit(main())
