"""Run the historical UNG adaptive trade-desk simulation."""
from __future__ import annotations

import json
from pathlib import Path

from ung_forecast.providers import fetch_alpaca_ung_daily
from ung_forecast.trade_desk import run_walk_forward_simulation


def main() -> None:
    fetched = fetch_alpaca_ung_daily()
    bars = fetched.frame
    report = run_walk_forward_simulation(bars, min_train_rows=252, step=5, max_origins=126)
    report["price_source"] = fetched.snapshot.provider
    report["data_through"] = fetched.snapshot.published_through.isoformat()
    output = Path("data/ledger/trade-desk-simulation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "bars", "simulation_origins", "first_origin", "last_origin",
        "metrics_by_horizon", "latest_factor_weights", "latest_forecasts",
    )}, indent=2))
    print(f"Source: {fetched.snapshot.provider} through {fetched.snapshot.published_through}")
    print(f"Full simulation saved to {output}")


if __name__ == "__main__":
    main()
