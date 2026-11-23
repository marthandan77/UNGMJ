# UNG Trade Desk: Product and Research Plan

## Objective

Produce calibrated, tradable UNG ranges for each of the next five sessions, explain the
forecast with point-in-time factor evidence, and continuously test new forecasting signals.
The system supports research and human-reviewed paper trading; it does not place live orders.

## Design decisions

1. Forecast UNG itself. UNG tracks a short-term natural-gas futures benchmark, rather than
   spot gas. The model therefore needs the NYMEX futures curve, contract mapping, and roll
   calendar alongside UNG prices. See the [fund benchmark description](https://www.uscfinvestments.com/ung).
2. Forecast distributions, not one low/high pair. For each of the next five sessions, estimate
   quantiles of the session low, close, and high relative to the previous close. Also retain a
   separate cumulative five-session path range for portfolio risk.
3. Separate range prediction from trading decisions. A range alone does not say that a limit
   order fills, that the low occurs before the high, or that a round trip is profitable.
4. Every input and model decision must have an `available_at` timestamp. Missing or late inputs
   are visibly unavailable and cannot silently become zeros or be backfilled from future data.

## Data plan

| Data | Use | Access and availability | Priority |
| --- | --- | --- | --- |
| UNG OHLCV, trades/quotes, intraday bars | Exact traded instrument, daily ranges, spread, liquidity, fill simulation | Alpaca API consolidated SIP daily and one-minute extended-hours bars; free-plan SIP data is accessible after 15 minutes. No quote depth feed is configured | P0 |
| NYMEX Henry Hub futures chain: front and deferred contracts, settlements, volume, open interest, bid/ask | Benchmark return, curve slope/calendar spreads, volatility, roll state | Exchange-quality history and real-time data generally require a licensed market-data provider; confirm rights and cost with provider/CME | P0 |
| UNG holdings, benchmark contract, scheduled roll | Align ETF movement with the actual contract exposure and roll dates | USCF publishes fund information and disclosures; capture dated snapshots | P0 |
| EIA weekly storage and other natural-gas balances | Surprise and seasonal balance features | EIA API is available; this repo already has storage ingestion. Use the actual release timestamp and construct an expectation using only prior data. [EIA API docs](https://www.eia.gov/opendata/documentation.php) | P1 |
| Weather forecasts for major US demand/supply regions | HDD/CDD revisions, forecast uncertainty, regional weighted demand shock | NWS exposes current grid and hourly forecasts. The system must archive each forecast vintage itself; current endpoints alone do not provide a historical as-issued forecast archive. [NWS API docs](https://www.weather.gov/documentation/services-web-api) | P1 |
| CFTC natural-gas positioning | Weekly managed-money and producer/merchant positioning changes | Free official CFTC API/archive; this repo's connector is now enabled in scheduled scans and the dashboard. Current historical availability is a conservative estimate from report date, so verify holiday-week vintages before treating it as exact. [CFTC COT data](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm) | P1 |
| LNG feedgas/export, dry-gas production, pipeline flows, outages | Physical supply/demand shocks | Identify a point-in-time source for each series; some data may require a paid provider. Do not use revised history as if first published. | P2 |
| Options implied volatility/skew, cross-asset prices, event calendar | Tail risk, market stress, conditional range and event effects | Licensed options data may be required; cross-asset bars can use existing market-data vendor if licensing allows | P2 |

Do not treat the list as permission to purchase a data subscription. Before connecting paid
feeds, compare license, redistribution limits, historical depth, latency, and cost. CME's
[market-data licensing guide](https://www.cmegroup.com/market-data/files/information-license-agreement-ila-guide.pdf)
describes its licensing framework. The initial build should use public data plus a clearly
marked provider interface for licensed futures and intraday data.

## Forecasting system

### Forecast targets

For each session `t+h`, `h = 1..5`, calculate targets relative to the previous close:

- low return: `log(L[t+h] / C[t+h-1])`
- close return: `log(C[t+h] / C[t+h-1])`
- high return: `log(H[t+h] / C[t+h-1])`

Model the 10th, 25th, 50th, 75th, and 90th percentiles. Enforce low ≤ close ≤ high and
consistent horizons. Train and score every horizon independently. The current cumulative
range forecast remains a separate risk view.

### Selected adaptive model

- **Single base learner:** HistGradientBoosting remains the only forecasting algorithm.
  It is fit as one full-feature expert and as feature-group experts (momentum, volatility/range,
  trend, volume, gas context, and outside-RTH once that feed is connected).
- **Dynamic Model Averaging (DMA):** combine the HGB experts separately for each forecast
  horizon. Start with equal weights, then update weights from matured out-of-sample low, close,
  and high errors using a forgetting factor. A weak expert loses weight gradually; it is not
  permanently removed after a short bad streak. The combined forecast is labeled `dma_consensus`.
- **Multi-horizon adaptive calibration:** maintain a separate calibration state for each
  horizon. ACI-style updates respond to whether the actual full-session range fell outside the
  forecast, while recent lower- and upper-bound miss sizes set the amount of expansion. A
  5-session forecast only updates its calibration after the full five-session outcome matures.
- **Future expert groups:** the outside-RTH UNG group is defined as a candidate input group;
  overnight NG inputs remain future work. Alpaca's one-minute SIP bars supply UNG premarket
  and prior-session after-hours OHLCV snapshots at 6:45 a.m. Eastern (premarket midpoint) and
  8:30 a.m. Eastern (one hour before RTH), stored separately going forward. Retrieval is scheduled
  15 minutes after each cutoff to respect free-plan SIP access. The collector does not provide
  quote depth or NG overnight bars, so no depth or
  futures-relative feature is included yet. The live expert waits for 105 snapshots to leave
  at least 100 mature training rows after the forecast-horizon embargo.
  Weather, LNG, production, and event-specific experts likewise remain unavailable until their
  timestamped feeds are implemented.
- **Range and drift monitoring:** report per-horizon coverage, width, low/high/close error,
  and component weights. Weight adaptation is evidence-driven and remains provisional until
  walk-forward performance supports it; it does not guarantee forecast accuracy.

The online multi-step calibration approach is informed by [Adaptive Conformal Inference for
Multi-Step Ahead Time-Series Forecasting Online](https://arxiv.org/abs/2409.14792). The choice
to keep the base learner simple while adapting expert weights is a project design decision,
not a claim that the method has already demonstrated an UNG trading edge.

Natural-gas distributional forecasting research motivates handling heavy tails and conditional
volatility in the forecast targets ([Berrisch & Ziel](https://arxiv.org/abs/2010.06227)). DMA
research motivates time-varying weights when the most useful expert can change over time
([Raftery, Kárný & Ettler](https://pmc.ncbi.nlm.nih.gov/articles/PMC2895940/)). These are
research leads, not evidence that the method will be profitable on UNG.

## Factor attribution and continual alpha research

Start with named factor families: futures curve/roll, recent returns and momentum, range and
realized volatility, trend, volume/liquidity, weather revisions, storage surprises, CFTC
position changes, and event flags. For each forecast, show:

- direction and size each family contributes to each predicted quantile;
- current expert weights and the matured sample count behind them;
- grouped ablation/permutation results from out-of-sample windows;
- stability across years, volatility regimes, and forecast horizons.

These are predictive contributions, not causal effects. A factor is promoted only if it improves
out-of-sample probabilistic scores and after-cost paper-trading results across multiple periods.
The alpha loop should create a candidate, log its data/feature version, run purged walk-forward
tests with a horizon embargo, compare against simple baselines, apply multiple-testing controls,
and keep or reject it. Automatically reduce a candidate's weight after drift or repeated
underperformance; never silently rewrite old forecasts.

## Trading layer for repeated buy-low/sell-high

The order planner should derive candidate limit prices from conditional quantiles, then estimate
the probability of a buy fill and a later sell fill using intraday paths. Rank setups by expected
net value after spread, commissions, slippage, and a missed-fill penalty. Enforce position,
inventory, daily-loss, and event-risk limits. OHLC-only backtests must not assume a favorable
intraday order of high and low; realistic fill logic requires intraday bars or quote data.

## Delivery sequence

1. **Data foundation:** persist raw vendor payloads, checksums, timestamps, contract identifiers,
   revisions, and data-quality flags; enable the existing CFTC connector; archive weather vintages;
   collect Alpaca premarket/after-hours snapshots; establish a futures provider for bid/ask,
   depth, contract-chain, and NG overnight features.
2. **Adaptive forecast:** keep HistGradientBoosting as the sole base learner; generate its
   full-feature and source-group experts; combine them with horizon-specific DMA and adaptive
   conformal range calibration.
3. **Trade Desk screens:** current price/futures curve, daily 1–5 range ladder, factor attribution,
   model weights, data freshness, upcoming events, range coverage, and probability of touching levels.
4. **Alpha lab:** scheduled candidate generation/evaluation, drift monitoring, versioned leaderboards,
   and human-reviewed promotion to paper trading.
5. **Paper execution:** intraday fill simulation with costs and risk limits. No live order routing
   until paper results are stable and the user explicitly requests execution integration.

## Promotion metrics

Track quantile loss and weighted interval score, coverage versus interval width, low/high/close
errors, calibration by horizon and regime, touch/fill precision, net P&L after costs, turnover,
drawdown, and performance versus a no-change/range baseline. A candidate must improve the metric
that matters for the intended trade without materially damaging tail coverage or risk.
