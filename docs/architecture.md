# Architecture and acceptance gates

## Target contract
The forecast target is the UNG ETF. Each forecast is issued at a recorded, timezone-aware cutoff and covers 1, 2, 3, or 5 subsequent trading sessions. Each horizon must produce probability of a positive return, expected return, implied expected close, predicted high/low range, and calibrated uncertainty quantiles.

## Data layers
1. Raw source snapshots: preserve provider identity, retrieval time, source URL, raw-content SHA-256, and provider publication/session coverage.
2. Normalized observations: validated daily UNG OHLCV and EIA Henry Hub daily spot prices. Preserve provenance and reject duplicate or conflicting sessions.
3. Point-in-time feature matrix: every feature must have been available by the forecast cutoff. A date alone does not prove availability for lagged-release data.
4. Targets: forward returns and future ranges generated only for training/evaluation; never included as features.
5. Immutable forecast ledger: append-only forecast vintages and separate one-time matured-outcome records.
6. Evaluation: chronological walk-forward predictions, per-horizon metrics, calibration, baselines, and ablation results.

## Source approval gate
A source is not approved merely because an endpoint responds. Validate the real instrument, meaningful historical depth, field coverage, session dates, duplicate/missing observations, corporate-action behavior where relevant, rate limits, free-tier restrictions, and terms. EIA Henry Hub spot data must be parsed into dated numeric observations. Current Henry Hub futures history is unresolved and remains disabled until independently validated. Weather is excluded.

## Modeling governance
- Ridge is the fixed production baseline.
- Challengers run in shadow evaluation only; no automatic model promotion.
- Compare all eligible history, rolling windows, and recency weighting via chronological walk-forward evaluation; no fixed 90-calendar-day window.
- Measure directional skill against a naive baseline, return error, range coverage/width, and probability calibration.
- Test incremental feature value with out-of-sample ablation and permutation tests. Do not select features on full-sample performance.
- Freeze each forecast before outcomes are observed. Score each matured vintage once; never backfill fictional historical forecasts.

## Publication and operations
The canonical GitHub Actions orchestrator collects and validates data, creates provisional Ridge forecasts, scores matured vintages once, runs chronological expanding/rolling/recency-weighted strategy comparisons and feature-group ablations, and builds a static dashboard artifact. The dashboard is presentation-only. When both UNG sources are available, they must be fresh, overlap for at least 500 sessions, agree on the latest session, and have median historical close differences within tolerance; a disagreement blocks the run. If secondary sources are unavailable, a valid fresh primary source may produce PROVISIONAL forecasts without any claim of independent confirmation. If the primary source or required EIA spot data is unavailable or stale, the run is BLOCKED. The scheduled workflow runs only after merge and persists ledger state on the separate `research-state` branch. A real-data end-to-end run is still required before operational readiness.
