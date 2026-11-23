# UNG Rolling Forecast

Professional research system for daily low/close/high forecasts across the next five trading sessions.

## Principles

- Point-in-time data only; no look-ahead.
- Fail closed on stale, malformed, incomplete, or unconfirmed data.
- HistGradientBoosting is the only base learner. It is fit as a full-feature model and as
  feature-group experts; all component forecasts and the final DMA forecast are recorded.
- Forecast vintages are immutable and scored only after their target session matures.
- Matured out-of-sample errors update horizon-specific Dynamic Model Averaging (DMA) weights.
  A multi-horizon adaptive conformal layer widens or tightens the DMA range from joint range misses.
- The system is research-only and does not execute trades.

## Development

```powershell
python -m pip install -e ".[dev]"
pytest
streamlit run streamlit_app.py
```

## Live data configuration

Set these environment variables in your local shell or deployment secret store; do not commit
credentials. For GitHub Actions, add `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` under the
repository's **Settings → Secrets and variables → Actions → New repository secret**.

- `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` are required for Alpaca US equity data. The
  collector requests consolidated SIP bars after the free feed's 15-minute access window.
- `EIA_API_KEY` enables official weekly working-gas storage ingestion.
- CFTC positioning uses the official annual archive automatically; `CFTC_NATURAL_GAS_CSV_URL` can override it with an official normalized CSV export.
- Missing context sources leave the run `PROVISIONAL`; they never silently become zeroes.

The Streamlit dashboard is organized as a trading desk, forecast ladder, learning view, market
input monitor, and walk-forward simulation. The scheduled GitHub Actions runner stores the
forecast and outcome ledgers under `data/ledger/`.

The scheduled runner refits HistGradientBoosting experts from the latest available daily
history. DMA uses only previously matured expert outcomes; the consensus is scored for
calibration and monitoring but excluded from the expert-weight calculation. Weights start
from an equal prior and adapt as evidence accumulates. This feedback loop does not guarantee
predictive accuracy.

## Adaptive Trade Desk simulation

See [`TRADE_DESK_PLAN.md`](TRADE_DESK_PLAN.md) for the full architecture, data-source map,
alpha research loop, and staged build plan.

Run a future-blind walk-forward simulation of the 1–5 session path range forecasts:

```powershell
$env:PYTHONPATH = "src"
python simulate_trade_desk.py
```

The simulation compares momentum, trend, volatility/range, and volume experts, all using
HistGradientBoosting. It learns horizon-specific expert weights from matured forecast errors
and calibrates range widths from prior out-of-sample path misses. The Streamlit dashboard
provides an interactive simulation panel and download. With current feeds, these factors are
derived from UNG OHLCV only. Displayed factor
contributions are model associations and should not be interpreted as causal effects.
The report includes comparison with a zero-return close forecast and empirical range
coverage so that adaptive performance can be judged against simple baselines.

The dashboard's **Market input availability** table shows the latest UNG provider checks,
EIA storage and CFTC positioning observations, plus the latest EIA Henry Hub spot value.
Henry Hub is reference-only until its historical publication times are verified. Market
depth and other unconfigured inputs are labeled as unavailable rather than treated as
forecast features.

Alpaca supplies UNG daily SIP bars and one-minute extended-session bars. The scheduled
collector fetches after the 15-minute delay and records the premarket midpoint (6:45 a.m.
Eastern) and the one-hour-before-open snapshot (8:30 a.m. Eastern). Both are stored per
target session in `data/ledger/outside-rth-features.jsonl`. These are consolidated trade bars,
not bid/ask quotes or order-book depth. The outside-RTH expert learns from snapshots
collected forward and stays out of the forecast until it has at least 105 usable snapshots
for its own time slot (100 training rows after the horizon embargo). The walk-forward
simulation activates it after at least 100 saved snapshots.

## GitHub storage controls

The scheduled ledger keeps at most 365 days and 400 rows per model for forecasts/outcomes,
and the most recent 1,200 outside-RTH snapshots. It compacts JSONL rows, does not make duplicate
monthly archives, and stops a scheduled update if tracked JSON data exceeds an 8 MiB budget.
Large simulation output is available as a download and is not committed to the repository.

Click **Refresh data & forecast** to fetch configured feeds, score forecasts whose target sessions have
completed, refit the HistGradientBoosting experts, and refresh the forecast, scorecard, feed
inventory, and run report. Market depth and other unconfigured sources remain unavailable
until providers are added.

## Validation

```powershell
$env:PYTHONPATH = "src"
pytest -q
python run_forecast.py
```
