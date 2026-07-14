# Engineering Rules

## Product boundary

Build one forecast-only UNG decision-support application. It must not place orders, connect to a broker, manage holdings, or present forecasts as guarantees.

## Forecast horizons

The application must maintain independent models for:

- 60 minutes
- 4 hours
- 1 complete trading session
- 2 complete trading sessions
- 7 complete trading sessions

## Quantitative rules

- Forecast outcomes are `LOWER_FIRST`, `UPPER_FIRST`, and `NEITHER`.
- Class probabilities must sum to one.
- Initial model family: multinomial elastic-net logistic regression.
- Advice must be derived from expected value and its conservative confidence bound, never from a fixed indicator or probability threshold.
- Training, validation, calibration, and test periods must remain separated.
- Use purged walk-forward validation with horizon-aware embargo.
- No random train/test split.
- No centered rolling windows or future information in features.
- All five horizon models use one canonical data policy and shared formula library, but independent fitted coefficients and calibration.
- Unvalidated horizons must show `RESEARCH STATUS ONLY`.

## Explanation rules

- Human-language explanations must be deterministic templates.
- Explanations may use only stored probabilities, expected values, validation metrics, data-health states, and stable model contributions.
- Never invent market causes, buyer or seller intent, institutional activity, news explanations, or guaranteed price targets.

## Learning rules

- The approved champion model remains stable.
- Background learning may train challenger models only from verified outcomes.
- A challenger cannot silently promote itself.
- Promotion requires predeclared validation criteria, shadow evaluation, versioning, and rollback metadata.

## Data rules

- Initial provider: `yfinance` for UNG and optional `NG=F`.
- Treat yfinance as prototype/research data.
- Use completed bars only.
- Reject stale, duplicate, malformed, or misaligned observations.
- Cache validated data with provenance.
- Support a separately trained fallback model without `NG=F`.

## Engineering rules

- Quantitative logic must remain outside Streamlit.
- Training and forecasting must call the same feature functions.
- Use typed Python and immutable schemas where practical.
- Every forecast records timestamp, horizon, data version, feature version, model version, configuration hash, probabilities, price areas, expected values, confidence, and validation status.
- Every requirement must map to implementation and tests.
- Do not add unrelated indicators, models, thresholds, or features without updating the build specification and acceptance criteria.
