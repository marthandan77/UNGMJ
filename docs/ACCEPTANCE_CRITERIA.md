# Mandatory Acceptance Criteria

The project is not approved merely because the Streamlit application starts. Every mandatory criterion below must be evidenced by code, automated tests, and a compliance report.

## Data

- AC-DATA-001: Only completed bars enter forecasting and training.
- AC-DATA-002: Duplicate timestamps are rejected.
- AC-DATA-003: Stale data suspends advice.
- AC-DATA-004: Timezone is normalized to America/New_York.
- AC-DATA-005: Data provenance and download time are recorded.
- AC-DATA-006: Adjusted-price policy is explicit and consistent.
- AC-DATA-007: NG=F failure uses a separately trained fallback model, not imputation into a model expecting the feature.

## Horizons

- AC-HOR-001: 60m means 12 completed 5-minute bars.
- AC-HOR-002: 4h means 16 completed 15-minute bars.
- AC-HOR-003: 1d means the next complete regular trading session.
- AC-HOR-004: 2d means two complete sessions.
- AC-HOR-005: 7d means seven complete sessions.
- AC-HOR-006: Each horizon has independent fitted model and calibration artifacts.

## Features and labels

- AC-FEAT-001: Training and forecasting call the same feature functions.
- AC-FEAT-002: Every feature records or can prove its maximum source timestamp is not after the prediction timestamp.
- AC-FEAT-003: Centered rolling calculations are prohibited.
- AC-LABEL-001: Labels are path-dependent LOWER_FIRST, UPPER_FIRST, or NEITHER.
- AC-LABEL-002: Barrier construction is identical between historical labeling and forecast-time interpretation.

## Models and validation

- AC-MODEL-001: Probabilities are finite, non-negative, and sum to one.
- AC-MODEL-002: Baseline model is multinomial elastic-net logistic regression.
- AC-VAL-001: Random train/test splits are prohibited.
- AC-VAL-002: Walk-forward folds are purged and horizon-aware embargoed.
- AC-VAL-003: Scalers, hyperparameters, and calibrators use training/validation data only.
- AC-VAL-004: Final test periods remain untouched until final evaluation.
- AC-VAL-005: Unvalidated horizons display RESEARCH STATUS ONLY.
- AC-VAL-006: Approved models beat declared base-rate baselines on probabilistic and economic criteria.

## Decisions

- AC-DEC-001: Sell advice is calculated from expected value and uncertainty, not an indicator or probability cutoff.
- AC-DEC-002: Buy advice is calculated from expected value and uncertainty.
- AC-DEC-003: Positive advice requires a positive conservative lower confidence bound.
- AC-DEC-004: Economically indeterminate forecasts output NO CLEAR ADVANTAGE — WAIT.

## Confidence and explanation

- AC-CONF-001: Confidence is derived from probability margin, distribution similarity, calibration quality, and perturbation stability.
- AC-EXP-001: All visible explanations use deterministic templates.
- AC-EXP-002: Every explanatory statement maps to stored model evidence.
- AC-EXP-003: Unsupported causal language is prohibited.

## Cross-horizon behavior

- AC-CROSS-001: Cross-horizon logic reports coherence and conflicts without altering individual probabilities.
- AC-CROSS-002: No arbitrary weighted average creates an overall buy/sell score.

## Learning

- AC-LEARN-001: Outcomes are scored only after the complete forecast window closes.
- AC-LEARN-002: Challenger training uses verified observations.
- AC-LEARN-003: A challenger cannot silently promote itself.
- AC-LEARN-004: Promotion requires declared metrics, shadow evaluation, versioning, and rollback metadata.

## Dashboard and operation

- AC-UI-001: Quantitative formulas remain outside Streamlit pages.
- AC-UI-002: All five horizons show status, probabilities, price areas, expected values, confidence, freshness, and model version.
- AC-UI-003: Invalid or stale data displays FORECAST UNAVAILABLE.
- AC-UI-004: Research-only outputs cannot be rendered as actionable guidance.

## Delivery

- AC-DEL-001: Requirement-to-code-to-test traceability is complete.
- AC-DEL-002: Automated tests pass.
- AC-DEL-003: Build compliance report lists any unresolved limitation honestly.
- AC-DEL-004: Codex audit is restricted to defects and compliance, not architecture redesign.
