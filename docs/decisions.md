# Decision log

## Accepted decisions
- Target: UNG ETF, not Henry Hub spot or futures.
- Horizons: 1, 2, 3, and 5 trading sessions.
- Weather: explicitly excluded.
- Data access: free public sources and free API keys only; private, non-commercial research.
- Baseline: Ridge regression; challengers shadow-only until reviewed.
- Training history: begin when enough valid history exists; compare all eligible history, rolling windows, and recency weighting rather than fixing a 90-day window.
- Validation: chronological walk-forward only; no random splits or look-ahead.
- Data integrity: fail closed on stale, incomplete, inconsistent, or misaligned data.
- Forecast ledger: immutable forecast vintages, with one-time outcome scoring after maturity.
- Workflow: one canonical orchestrator when introduced; dashboard is presentation-only; no email delivery, broker integration, or trade execution.

## Not yet approved
- UNG market-data provider.
- EIA workbook/API parser and normalized history.
- Current Henry Hub futures provider and history.
- Storage or CFTC data integration.
- Production forecast thresholds, uncertainty calibration method, or scheduling cadence.
- Any model challenger promotion.
