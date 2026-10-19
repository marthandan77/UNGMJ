"""Rolling five-session individual-model forecast engine."""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .features import build_features, build_forward_targets
from .calibration import calibrate_range
from .ledger import ForecastLedger
from .models import fit_individual_forecast
from .sessions import next_sessions
from .weights import build_adaptive_consensus, compute_model_weights


def generate_rolling_forecasts(
    ung: pd.DataFrame,
    *,
    henry_hub: pd.Series | pd.DataFrame | None = None,
    context: pd.DataFrame | None = None,
    model_names: Iterable[str] | None = None,
    ledger: ForecastLedger | None = None,
    min_rows: int = 100,
) -> list[dict]:
    """Generate one independent forecast per model and future session."""
    ung = ung.sort_index()
    if model_names is None:
        configured = os.getenv("UNG_MODELS", "ridge")
        model_names = tuple(name.strip() for name in configured.split(",") if name.strip())
    model_names = tuple(dict.fromkeys(model_names))
    features = build_features(ung, henry_hub, context)
    latest_session = ung.index[-1]
    issued_at = datetime.now(timezone.utc).isoformat()
    output: list[dict] = []
    future_sessions = next_sessions(latest_session, 5)
    for horizon in (1, 2, 3, 4, 5):
        target_session = future_sessions[horizon - 1].date().isoformat()
        targets = build_forward_targets(ung, horizon)
        for model_name in model_names:
            forecast = fit_individual_forecast(
                model_name, features, targets, reference_close=float(ung.iloc[-1].close),
                target_session=target_session, horizon=horizon, min_rows=min_rows,
            )
            identity = f"{latest_session.date()}|{target_session}|{model_name}|{horizon}"
            record = {
                "forecast_id": "UNG-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
                "issued_at": issued_at,
                "origin_session": latest_session.date().isoformat(),
                "target_session": forecast.target_session,
                "horizon": forecast.horizon,
                "model_name": forecast.model_name,
                "reference_close": float(ung.iloc[-1].close),
                "predicted_low": forecast.predicted_low,
                "expected_close": forecast.expected_close,
                "predicted_high": forecast.predicted_high,
                "probability_up": forecast.probability_up,
                "uncertainty": forecast.uncertainty,
                "status": "PROVISIONAL",
            }
            output.append(record)
            if ledger is not None:
                record = calibrate_range(record, ledger.outcomes())
                output[-1] = record
            if ledger is not None:
                ledger.append_forecast(record)
    if ledger is not None:
        # Only matured outcomes from earlier forecasts can influence today's
        # adaptive consensus. Every individual model forecast remains visible.
        learned_weights = compute_model_weights(ledger.outcomes())
        for consensus in build_adaptive_consensus(output, learned_weights):
            identity = (
                f"{consensus['origin_session']}|{consensus['target_session']}|"
                f"{consensus['model_name']}|{consensus['horizon']}"
            )
            consensus["forecast_id"] = "UNG-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            consensus["issued_at"] = issued_at
            consensus = calibrate_range(consensus, ledger.outcomes())
            output.append(consensus)
            ledger.append_forecast(consensus)
    return output
