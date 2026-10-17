"""Scheduled entry point for one validated rolling forecast run."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from ung_forecast.ledger import ForecastLedger
from ung_forecast.providers import fetch_nasdaq_ung, fetch_yahoo_ung, compare_sources
from ung_forecast.context_data import combine_context_sources, fetch_cftc_positioning, fetch_eia_storage
from ung_forecast.rolling import generate_rolling_forecasts
from ung_forecast.weights import compute_model_weights


def main() -> None:
    fetched = fetch_nasdaq_ung()
    yahoo = fetch_yahoo_ung()
    agreement = compare_sources(fetched, yahoo)
    context = []
    context_status = {}
    for name, fetcher in (("eia_storage", fetch_eia_storage), ("cftc_positioning", fetch_cftc_positioning)):
        try:
            context.append(fetcher())
            context_status[name] = "AVAILABLE"
        except Exception as exc:
            context_status[name] = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    ledger = ForecastLedger()
    features_context = combine_context_sources(context)
    configured_models = tuple(
        name.strip() for name in os.getenv(
            "UNG_MODELS", "ridge,hist_gradient_boosting,random_forest,extra_trees"
        ).split(",") if name.strip()
    )
    rows = generate_rolling_forecasts(
        fetched.frame, context=features_context, ledger=ledger,
        model_names=configured_models,
    )
    scored = ledger.score_matured(fetched.frame)
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "status": "READY" if len(context) == 2 else "PROVISIONAL",
        "provider": fetched.snapshot.provider,
        "agreement": agreement,
        "context": context_status,
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
