# UNG Forecast Machine — Locked Build Specification

## 1. Objective

Build a human-readable, forecast-only application for UNG. For each approved horizon, estimate whether a meaningful lower price, meaningful higher price, or neither event is reached first. Convert calibrated probabilities and conditional movement estimates into expected-value-based guidance for a user considering selling or buying.

## 2. Horizons

- `60m`: next 12 completed 5-minute bars.
- `4h`: next 16 completed 15-minute bars.
- `1d`: through the close of the next complete regular trading session.
- `2d`: through the close of the second complete regular trading session.
- `7d`: through the close of the seventh complete regular trading session.

Each horizon has independent labels, feature specification, scaler, fitted coefficients, calibrator, validation report, champion, and challenger metadata.

## 3. Outcomes

For horizon H:

`Y_H ∈ {LOWER_FIRST, UPPER_FIRST, NEITHER}`

`P_H(LOWER_FIRST) + P_H(UPPER_FIRST) + P_H(NEITHER) = 1`

The target is path-dependent. It is not merely the sign of the final return.

## 4. Price barriers

Transparent baseline:

`B_lower,H = P_t - k_lower,H × sigma_hat_H,t`

`B_upper,H = P_t + k_upper,H × sigma_hat_H,t`

Barrier multipliers are selected only within training/validation folds. Untouched test observations cannot influence them. Future challengers may estimate conditional movement quantiles directly.

## 5. Shared formula library

All horizons use central implementations of:

- Log return: `r_t,w = ln(P_t / P_t-w)`
- Realized volatility: `RV_t,w = sqrt(sum(r_i^2))`
- Session VWAP using observations available at time t only
- Standardized VWAP deviation
- Variance ratio
- Time-adjusted relative volume
- Rolling UNG–NG return residual, where validated
- Trading-calendar and time encodings

Training and live forecasting must import the same functions.

## 6. Baseline model

Each horizon starts with multinomial elastic-net logistic regression. Probabilities must be calibrated using data separated from final testing.

No HMM, GARCH, tree ensemble, neural network, or heuristic indicator voting enters the approved baseline unless independently tested later as a challenger.

## 7. Decision engine

For selling:

`EV_sell,H = p_lower × E[decline | lower] - p_upper × E[upside | upper] - costs - uncertainty_penalty`

For buying:

`EV_buy,H = p_upper × E[upside | upper] - p_lower × E[decline | lower] - costs - uncertainty_penalty`

Positive advice requires the conservative lower confidence bound of the relevant expected value to exceed zero. If statistically indistinguishable from zero, output `NO CLEAR ADVANTAGE — WAIT`.

No fixed indicator threshold or fixed class-probability threshold may directly create advice.

## 8. Confidence

Confidence must be derived from model evidence:

`confidence = probability_margin × in_distribution_similarity × calibration_quality × perturbation_stability`

Human labels are mapped from validation-derived bands. If data is materially out of distribution, stale, malformed, or the horizon is unvalidated, directional advice is suspended.

## 9. Cross-horizon coordinator

For each horizon:

`mu_H = P_H(UPPER_FIRST) - P_H(LOWER_FIRST)`

Adjacent-horizon coherence is monitored from changes in `mu_H`. The coordinator summarizes term structure and disagreement but never overwrites a horizon model or averages the five forecasts into an arbitrary overall score.

## 10. Explanations

Visible language is generated only from deterministic templates populated with:

- probabilities,
- expected values,
- conditional price areas,
- data health,
- validation metrics,
- stable model contributions.

Unsupported causal statements are prohibited.

## 11. Data policy

Initial provider: Yahoo Finance via `yfinance`.

Symbols:

- `UNG`
- Optional `NG=F`

Requirements:

- completed bars only,
- America/New_York normalization,
- duplicate rejection,
- stale-data detection,
- provenance and download timestamps,
- explicit adjusted-price policy,
- Parquet cache,
- fallback horizon models trained without `NG=F`.

Insufficient sample size or failed validation produces `RESEARCH STATUS ONLY` rather than fabricated advice.

## 12. Validation

Use purged walk-forward evaluation with horizon-aware embargo. Prohibit random train/test splits.

Required comparisons:

- unconditional class-frequency baseline,
- recency-weighted class-frequency baseline,
- plain multinomial logistic regression,
- elastic-net multinomial logistic regression.

Required metrics:

- multiclass Brier score,
- log loss,
- expected calibration error,
- class reliability,
- barrier timing and coverage,
- expected-value results after spread/slippage assumptions,
- missed-upside and premature-buy costs,
- stability across periods and feature ablations.

## 13. Background learning

Every forecast is stored and scored only after its complete horizon closes. Verified observations may train a challenger. The challenger runs in shadow mode and may not replace the champion without passing predeclared validation, stability, and rollback requirements.

## 14. Dashboard

Pages:

1. Current Forecast
2. Forecast History
3. Model Performance
4. System Health

For each horizon show:

- validation status,
- human-readable advice,
- current price,
- estimated lower and upper areas,
- three outcome probabilities,
- sell expected value,
- buy expected value,
- confidence,
- calibration history,
- data freshness,
- model and feature versions.

Streamlit is a presentation and workflow layer. It must not contain quantitative formulas or alter forecast outputs.

## 15. Prohibited behaviour

- automatic order execution,
- invented market explanations,
- guaranteed forecasts,
- arbitrary technical gates,
- free-form LLM reasoning in the forecast path,
- random splits,
- centered rolling features,
- silent online model replacement,
- hidden post-test tuning,
- one fitted model reused for all horizons.
