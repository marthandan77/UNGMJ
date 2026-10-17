# UNG Rolling Forecast

Professional research system for daily low/close/high forecasts across the next five trading sessions.

## Principles

- Point-in-time data only; no look-ahead.
- Fail closed on stale, malformed, incomplete, or unconfirmed data.
- Every model produces an individual forecast and scorecard.
- Forecast vintages are immutable and scored only after their target session matures.
- The system is research-only and does not execute trades.

## Development

```powershell
python -m pip install -e ".[dev]"
pytest
streamlit run streamlit_app.py
```

## Live data configuration

Copy `.env.example` into your deployment secret store; do not commit credentials.

- `EIA_API_KEY` enables official weekly working-gas storage ingestion.
- CFTC positioning uses the official annual archive automatically; `CFTC_NATURAL_GAS_CSV_URL` can override it with an official normalized CSV export.
- Missing context sources leave the run `PROVISIONAL`; they never silently become zeroes.

The Streamlit dashboard allows selecting individual models. The scheduled GitHub Actions
runner evaluates all configured challenger models and persists immutable forecast/outcome
ledgers under `data/ledger/`.

## Validation

```powershell
$env:PYTHONPATH = "src"
pytest -q
python run_forecast.py
```
