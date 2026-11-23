"""Rolling five-session HistGradientBoosting and DMA forecast engine."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .calibration import calibrate_range
from .features import build_features, build_forward_targets, forecast_expert_columns
from .ledger import ForecastLedger
from .models import fit_individual_forecast
from .sessions import next_sessions
from .weights import build_dma_consensus, compute_model_weights


def generate_rolling_forecasts(
    ung: pd.DataFrame,
    *,
    henry_hub: pd.Series | pd.DataFrame | None = None,
    context: pd.DataFrame | None = None,
    outside_rth: pd.DataFrame | None = None,
    model_names: Iterable[str] | None = None,
    ledger: ForecastLedger | None = None,
    include_consensus: bool = True,
    min_rows: int = 100,
) -> list[dict]:
    """Forecast five horizons using HGB feature experts and their DMA blend."""
    if model_names is not None and any(name != "hist_gradient_boosting" for name in model_names):
        raise ValueError("HistGradientBoosting is the only supported base model")
    ung = ung.sort_index()
    features = build_features(ung, henry_hub, context, outside_rth)
    experts = forecast_expert_columns(features, min_observations=min_rows + 5)
    latest_session = ung.index[-1]
    issued_at = datetime.now(timezone.utc).isoformat()
    reference = float(ung.iloc[-1].close)
    output: list[dict] = []
    future_sessions = next_sessions(latest_session, 5)

    for horizon in (1, 2, 3, 4, 5):
        target_session = future_sessions[horizon - 1].date().isoformat()
        targets = build_forward_targets(ung, horizon)
        horizon_candidates: list[dict] = []
        for expert_name, columns in experts.items():
            forecast = fit_individual_forecast(
                "hist_gradient_boosting", features, targets,
                reference_close=reference, target_session=target_session,
                horizon=horizon, min_rows=min_rows, feature_columns=columns,
                expert_name=expert_name,
            )
            if not (0.25 * reference < forecast.predicted_low <= forecast.expected_close <= forecast.predicted_high < 4.0 * reference):
                raise ValueError(f"{expert_name} produced an unreasonable forecast for horizon {horizon}")
            identity = f"{issued_at}|{latest_session.date()}|{target_session}|{expert_name}|{horizon}"
            record = {
                "forecast_id": "UNG-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                "issued_at": issued_at,
                "origin_session": latest_session.date().isoformat(),
                "target_session": forecast.target_session,
                "horizon": forecast.horizon,
                "model_name": forecast.model_name,
                "algorithm": "HistGradientBoostingRegressor",
                "expert": expert_name,
                "reference_close": reference,
                "predicted_low": forecast.predicted_low,
                "expected_close": forecast.expected_close,
                "predicted_high": forecast.predicted_high,
                "probability_up": forecast.probability_up,
                "uncertainty": forecast.uncertainty,
                "status": "PROVISIONAL",
            }
            horizon_candidates.append(record)
            output.append(record)
            if ledger is not None:
                ledger.append_forecast(record)

        if include_consensus:
            learned_weights = compute_model_weights(ledger.outcomes()) if ledger is not None else []
            consensus_rows = build_dma_consensus(horizon_candidates, learned_weights)
            for consensus in consensus_rows:
                identity = (
                    f"{issued_at}|{consensus['origin_session']}|{consensus['target_session']}|"
                    f"{consensus['model_name']}|{consensus['horizon']}"
                )
                consensus["forecast_id"] = "UNG-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
                consensus["issued_at"] = issued_at
                if ledger is not None:
                    consensus = calibrate_range(consensus, ledger.outcomes())
                    ledger.append_forecast(consensus)
                else:
                    consensus["calibration_status"] = "NO_LEDGER_HISTORY"
                    consensus["calibration_sample_count"] = 0
                output.append(consensus)
    return output
