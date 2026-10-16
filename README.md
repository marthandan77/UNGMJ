# UNGMJ — UNG ETF Research Forecasting

Private-use, non-commercial research project for probabilistic forecasts of the United States Natural Gas Fund (UNG) ETF over 1, 2, 3, and 5 trading sessions.

## Scope
- Forecast target: UNG ETF prices and returns. Henry Hub spot/futures are explanatory inputs, never substitutes for the UNG target.
- Outputs: direction probability, expected return and price, forecast high–low range, and calibrated uncertainty.
- Required external input: EIA daily Henry Hub spot price, after its history is parsed and validated.
- Optional inputs: current/history Henry Hub futures, storage, CFTC positioning, and other public market data only after source coverage, terms, historical availability, and point-in-time timestamps are verified.
- Excluded: weather data, paid data plans, broker integrations, order execution, and trading instructions.
- Free access only: public sources and free personal-use API keys. Secrets must never be committed.

## Research discipline
1. Fail closed on malformed, stale, incomplete, or misaligned data.
2. Never use a feature before its actual publication/availability time.
3. Use chronological walk-forward validation only; never random-split time series.
4. Store immutable forecast vintages before the outcome period and score each vintage once when its target matures. Never fabricate historical forecasts.
5. Begin forecasts as soon as enough valid UNG history exists. Do not impose a fixed 90-calendar-day training window. Compare all-eligible-history, rolling-window, and recency-weighted approaches using walk-forward evidence.
6. Ridge regression is the fixed production baseline. Other models remain shadow challengers until explicitly reviewed; no automatic promotion.
7. Learn feature value from chronological out-of-sample ablation/permutation tests and stable importance analysis, not assumptions or in-sample fit.
8. If only one provider supplies a usable latest session, label the result PROVISIONAL; do not claim independent confirmation.
9. Never weaken data, validation, or forecast integrity checks merely to produce a forecast.

## Build status
The implementation includes contracts, candidate free-source ingestion (Nasdaq historical OHLCV first, Yahoo chart and Stooq as fallback/confirmation; official EIA API or public XLS for Henry Hub spot), fail-closed provider checks, point-in-time features, a Ridge baseline, chronological strategy/feature ablation evaluation, immutable forecast and outcome ledgers, a static read-only dashboard, a password-gated Streamlit dashboard, CI, and one scheduled orchestrator.
Forecasts remain **PROVISIONAL** until live feed reliability and out-of-sample uncertainty calibration are independently demonstrated. A live-data workflow run has succeeded using Nasdaq UNG OHLCV and the official EIA XLS history. Independent UNG confirmation remains unavailable in the latest run, so outputs stay PROVISIONAL. The Streamlit dashboard is deployed at https://ungmj-research-dashboard.onrender.com and uses a host-only `APP_PASSWORD` (never commit the password). The free host may sleep while idle. Important: because this repository and its `research-state` branch are public, the underlying forecast and ledger files are publicly readable; the dashboard password gates the UI only, not the raw files.

## Development
Python 3.12+. Install development dependencies and run tests:

    python -m pip install -e ".[dev]"
    pytest

The scheduled workflow runs at 02:30 UTC Tuesday–Saturday, restores and persists its ledger on the `research-state` branch, and publishes a 30-day workflow artifact containing the read-only dashboard and report. It begins only after the pull request is merged into `main`. A blocked run records the failure rather than manufacturing forecasts.
