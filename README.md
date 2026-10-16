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
The initial implementation establishes reusable data contracts and validation tests. No data provider, forecast, or model is approved yet. The EIA Henry Hub spot parser and UNG OHLCV providers must pass real historical-data validation before modeling begins.

## Development
Python 3.12+. Install development dependencies and run tests:

    python -m pip install -e ".[dev]"
    pytest

The first milestone intentionally has no scheduled workflow and does not fetch external data. Provider and orchestration workflows will be introduced after their contracts and tests are reviewed.
