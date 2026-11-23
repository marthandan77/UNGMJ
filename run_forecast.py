"""Scheduled entry point for one validated rolling forecast run."""
import json
import os
import time
from datetime import datetime, time as local_time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from ung_forecast.ledger import ForecastLedger
from ung_forecast.providers import fetch_alpaca_ung_daily, fetch_alpaca_ung_intraday
from ung_forecast.outside_rth import build_current_outside_rth_snapshot, outside_rth_frame
from ung_forecast.context_data import (
    combine_context_sources,
    fetch_cftc_positioning,
    fetch_eia_storage,
)
from ung_forecast.feed_status import (
    describe_context_feed,
    describe_price_feed,
    unavailable_feed,
)
from ung_forecast.rolling import generate_rolling_forecasts
from ung_forecast.weights import compute_model_weights


def main() -> None:
    started = time.perf_counter()
    if os.getenv("GITHUB_EVENT_NAME") == "schedule":
        now_et = datetime.now(ZoneInfo("America/New_York"))
        slot_windows = (
            (local_time(7, 0), local_time(7, 25)),
            (local_time(8, 45), local_time(9, 5)),
        )
        if not any(start <= now_et.time() <= end for start, end in slot_windows):
            print(f"Skipping non-study schedule at {now_et.isoformat()}")
            return
    fetched = fetch_alpaca_ung_daily()
    agreement = "Alpaca consolidated SIP"
    data_feeds = [
        describe_price_feed("Alpaca UNG daily SIP", fetched, "Primary UNG OHLCV"),
    ]
    context = []
    context_status = {}
    feed_specs = (
        (
            "eia_storage",
            "EIA storage",
            fetch_eia_storage,
            ("storage_bcf",),
            "https://www.eia.gov/dnav/ng/NG_STOR_WKLY_S1_W.htm",
            "Current release timestamp comes from the workbook/API rule. Historical workbook rows inherit the current workbook release date, so they are not point-in-time history.",
        ),
        (
            "cftc_positioning",
            "CFTC natural-gas positioning",
            fetch_cftc_positioning,
            ("managed_money_net_pct_oi", "producer_merchant_net_pct_oi"),
            "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
            "Weekly positions use a conservative release-time estimate; verify holiday-week release vintages before using in research backtests.",
        ),
    )
    for key, name, fetcher, value_columns, source_url, availability_note in feed_specs:
        try:
            frame = fetcher()
            context.append(frame)
            context_status[key] = "AVAILABLE"
            data_feeds.append(describe_context_feed(
                name, frame, value_columns=value_columns, availability_note=availability_note,
                source_url=source_url,
            ))
        except Exception as exc:
            context_status[key] = f"UNAVAILABLE: {type(exc).__name__}"
            data_feeds.append(unavailable_feed(name, f"Fetch or validation failed ({type(exc).__name__})."))
    data_feeds.extend([
        unavailable_feed(
            "Market depth", "No depth feed is configured in the daily UNG collector.", status="NOT_CONNECTED"
        ),
        unavailable_feed(
            "Weather / LNG / production", "No point-in-time provider is configured for these inputs.",
            status="NOT_CONFIGURED",
        ),
    ])
    ledger = ForecastLedger()
    outside_rth_slot = None
    try:
        intraday = fetch_alpaca_ung_intraday()
        snapshot = build_current_outside_rth_snapshot(intraday.frame, fetched.frame)
        snapshot["retrieved_at"] = intraday.snapshot.retrieved_at.isoformat()
        ledger.append_outside_rth_feature(snapshot)
        outside_rth_slot = snapshot["snapshot_slot"]
        outside_rth_status = "AVAILABLE"
        slot_prefix = "mid" if snapshot["snapshot_slot"] == "mid" else "1h"
        premarket_return = float(snapshot[f"ung_premarket_{slot_prefix}_return"])
        data_feeds.append({
            "source": "Alpaca UNG 1-minute extended session (SIP)",
            "status": outside_rth_status,
            "latest_observation": snapshot["target_session"],
            "latest_value": f"{snapshot['snapshot_slot']} premarket return {premarket_return:.3%}",
            "available_at": snapshot["available_at"],
            "retrieved_at": snapshot["retrieved_at"],
            "source_url": intraday.snapshot.source_url,
            "role": f"{snapshot['snapshot_slot']} pre-open snapshot and prior after-hours features",
            "note": "15-minute-delayed consolidated SIP OHLCV bars; no bid/ask spread or quote depth.",
        })
    except Exception as exc:
        outside_rth_status = f"UNAVAILABLE: {type(exc).__name__}"
        data_feeds.append(unavailable_feed(
            "Alpaca UNG 1-minute extended session (SIP)",
            f"Pre-open snapshot unavailable ({type(exc).__name__}).",
        ))
    features_context = combine_context_sources(context)
    outside_features = outside_rth_frame(ledger.outside_rth_features())
    # Score completed target sessions before training this run's adaptive layer.
    scored = ledger.score_matured(fetched.frame)
    rows = generate_rolling_forecasts(
        fetched.frame, context=features_context, outside_rth=outside_features, ledger=ledger,
    )
    purge = ledger.purge()
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "trigger": "scheduled_run",
        "status": "READY" if context else "PROVISIONAL",
        "provider": fetched.snapshot.provider,
        "agreement": agreement,
        "context": context_status,
        "outside_rth": outside_rth_status,
        "outside_rth_slot": outside_rth_slot,
        "data_feeds": data_feeds,
        "latest_session": str(fetched.frame.index[-1].date()),
        "history_rows": len(fetched.frame),
        "forecasts": len(rows),
        "scored": scored,
        "ledger_forecasts": len(ledger.forecasts()),
        "ledger_outcomes": len(ledger.outcomes()),
        "purge": purge,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "model_weights": compute_model_weights(ledger.outcomes()),
    }
    Path("data/ledger").mkdir(parents=True, exist_ok=True)
    Path("data/ledger/latest-run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
