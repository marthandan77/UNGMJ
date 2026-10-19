"""Scheduled entry point for one validated rolling forecast run."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from ung_forecast.ledger import ForecastLedger
from ung_forecast.providers import fetch_nasdaq_ung, fetch_yahoo_ung, compare_sources
from ung_forecast.context_data import (
    combine_context_sources,
    fetch_cftc_positioning,
    fetch_eia_henry_hub_spot,
    fetch_eia_storage,
)
from ung_forecast.feed_status import (
    describe_context_feed,
    describe_henry_hub_feed,
    describe_price_feed,
    unavailable_feed,
)
from ung_forecast.rolling import generate_rolling_forecasts
from ung_forecast.weights import compute_model_weights


def main() -> None:
    fetched = fetch_nasdaq_ung()
    yahoo = fetch_yahoo_ung()
    agreement = compare_sources(fetched, yahoo)
    data_feeds = [
        describe_price_feed("Nasdaq UNG", fetched, "Primary UNG OHLCV"),
        describe_price_feed("Yahoo UNG", yahoo, "Independent price confirmation"),
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
            "https://www.cftc.gov/dea/futures/nat_gas_lf.htm",
            "Availability uses the collector's conservative report-date + 4 days, 21:00 UTC rule; the archive does not provide exact historical publication timestamps.",
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
    try:
        henry_hub = fetch_eia_henry_hub_spot()
        data_feeds.append(describe_henry_hub_feed(henry_hub))
    except Exception as exc:
        data_feeds.append(unavailable_feed(
            "EIA Henry Hub spot", f"Fetch or validation failed ({type(exc).__name__})."
        ))
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
    features_context = combine_context_sources(context)
    configured_models = tuple(
        name.strip() for name in os.getenv(
            "UNG_MODELS", "ridge,hist_gradient_boosting,random_forest,extra_trees"
        ).split(",") if name.strip()
    )
    # Score completed target sessions before training this run's adaptive layer.
    scored = ledger.score_matured(fetched.frame)
    rows = generate_rolling_forecasts(
        fetched.frame, context=features_context, ledger=ledger,
        model_names=configured_models,
    )
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "status": "READY" if len(context) == 2 else "PROVISIONAL",
        "provider": fetched.snapshot.provider,
        "agreement": agreement,
        "context": context_status,
        "data_feeds": data_feeds,
        "latest_session": str(fetched.frame.index[-1].date()),
        "history_rows": len(fetched.frame),
        "forecasts": len(rows),
        "scored": scored,
        "ledger_forecasts": len(ledger.forecasts()),
        "ledger_outcomes": len(ledger.outcomes()),
        "model_weights": compute_model_weights(ledger.outcomes()),
    }
    Path("data/ledger").mkdir(parents=True, exist_ok=True)
    Path("data/ledger/latest-run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
