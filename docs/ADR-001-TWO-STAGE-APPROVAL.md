# ADR-001 — Two-Stage Approval and Shadow Mode

## Decision

Model approval is separated into two independent stages.

### Stage 1: Statistical approval

A horizon must demonstrate, through purged walk-forward testing:

- sufficient sample size,
- lower multiclass Brier score than the declared baseline,
- lower log loss than the declared baseline,
- acceptable expected calibration error,
- stability across all required folds.

Passing this stage permits a horizon to be described as statistically approved. It does not establish trading value.

### Stage 2: Trading approval

Trading approval additionally requires:

- a versioned execution model,
- explicit spread, commission, slippage, timing, and fill assumptions,
- positive out-of-sample economic value,
- a minimum number of scored shadow forecasts,
- no failure of the statistical approval stage.

Without these items, the horizon remains research-only or shadow-only.

## Shadow mode

Shadow mode generates and scores forecasts but cannot execute orders. The project remains forecast-only, and no component exposes an order-execution method.

## Forecast ledger

Forecast records are immutable and append-only. Each record includes model, feature, data, and configuration identity, advice, expiry, and verified outcome when available.

## Nightly report

A deterministic report summarizes forecast counts, pending/scored/invalid outcomes, directional accuracy, confidence, data-health failures, and model versions. It must not claim causal explanations or economic performance without the required evidence.

## Consequence

Statistical skill and trading profitability can no longer be conflated. This prevents an unconfigured or unrealistic execution assumption from producing a false validation claim.
