"""Walk-forward, factor-grouped adaptive range forecasting for UNG.

Factor contributions are model-based associations. They are not causal estimates.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

from .features import build_features, build_forward_targets


FACTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "momentum": ("return_1", "return_3", "return_5", "return_10", "return_20", "rsi_14"),
    "volatility_range": ("range_5", "range_10", "range_20", "realized_vol_5", "realized_vol_10", "realized_vol_20", "atr_14_pct"),
    "trend": ("price_vs_ema_5", "price_vs_ema_10", "price_vs_ema_20", "price_vs_ema_50"),
    "volume": ("volume_ratio_20", "volume_zscore_20"),
    "gas_context": ("henry_hub_return_1", "ung_gas_spread_1", "storage_bcf", "managed_money_net_pct_oi", "producer_merchant_net_pct_oi"),
    "outside_rth_mid": ("ung_after_hours_return", "ung_after_hours_range_pct", "ung_after_hours_volume_ratio", "ung_premarket_mid_return", "ung_premarket_mid_range_pct", "ung_premarket_mid_volume_ratio"),
    "outside_rth_1h": ("ung_after_hours_return", "ung_after_hours_range_pct", "ung_after_hours_volume_ratio", "ung_premarket_1h_return", "ung_premarket_1h_range_pct", "ung_premarket_1h_volume_ratio"),
}


def _learner() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingRegressor(
            max_iter=35, learning_rate=0.06, max_leaf_nodes=7,
            l2_regularization=4.0, random_state=42,
        ),
    )


def _adaptive_weights(history: list[dict[str, Any]], horizon: int, names: list[str]) -> dict[str, float]:
    recent: dict[str, list[float]] = defaultdict(list)
    for row in history:
        if row["horizon"] == horizon:
            recent[row["factor"]].append(float(row["loss"]))
    eligible = {name: values[-60:] for name, values in recent.items() if len(values) >= 10}
    raw = {name: 1.0 / (float(np.mean(values)) + 1e-4) for name, values in eligible.items()}
    # Until enough matured evidence exists, distribute weight equally among
    # factors; thereafter blend learned performance with a small equal prior.
    if not raw:
        return {name: 1.0 / len(names) for name in names}
    total = sum(raw.values())
    weights = {name: 0.15 / len(names) for name in names}
    for name in names:
        weights[name] += 0.85 * raw.get(name, 0.0) / total
    return weights


def _calibration_offsets(range_history: dict[int, dict[str, list[float]]], horizon: int) -> tuple[float, float]:
    errors = range_history.get(horizon, {})
    joint_errors = errors.get("joint", [])[-80:]
    # Calibrate the maximum of the lower- and upper-tail misses, so the
    # empirical target is joint path coverage rather than two marginal tails.
    offset = float(np.quantile(joint_errors, 0.90)) if len(joint_errors) >= 10 else 0.0
    return max(0.0, offset), max(0.0, offset)


def _predict_origin(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    origin: int,
    horizon: int,
    *,
    min_train_rows: int,
    history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    names = [name for name, columns in FACTOR_GROUPS.items() if any(c in features for c in columns)]
    weights = _adaptive_weights(history, horizon, names)
    # A target is usable only once all horizon sessions have elapsed.
    train_end = origin - horizon + 1
    if train_end <= 0:
        return None
    train = features.iloc[:train_end].join(targets.iloc[:train_end]).dropna(
        subset=["forward_low_return", "forward_return", "forward_high_return"]
    )
    if len(train) < min_train_rows:
        return None
    expert_predictions: dict[str, dict[str, float]] = {}
    for name in names:
        columns = [column for column in FACTOR_GROUPS[name] if column in features.columns]
        usable = [column for column in columns if train[column].notna().any()]
        usable_rows = int(train[usable].notna().any(axis=1).sum()) if usable else 0
        if not usable or usable_rows < min(100, min_train_rows):
            continue
        expert_predictions[name] = {}
        current = features.iloc[[origin]][usable]
        for output, target in (("low", "forward_low_return"), ("close", "forward_return"), ("high", "forward_high_return")):
            model = _learner()
            model.fit(train[usable], train[target])
            expert_predictions[name][output] = float(model.predict(current)[0])
    if not expert_predictions:
        return None
    weights = {name: weights[name] for name in expert_predictions}
    total = sum(weights.values())
    weights = {name: value / total for name, value in weights.items()}
    blended = {
        key: sum(weights[name] * prediction[key] for name, prediction in expert_predictions.items())
        for key in ("low", "close", "high")
    }
    reference = float(features.attrs["close"].iloc[origin])
    low = reference * (1 + blended["low"])
    close = reference * (1 + blended["close"])
    high = reference * (1 + blended["high"])
    low, high = min(low, close, high), max(low, close, high)
    attribution = [
        {"factor": name, "weight": weights[name], "predicted_close_contribution_pct": 100 * weights[name] * prediction["close"]}
        for name, prediction in expert_predictions.items()
    ]
    return {
        "origin_index": origin,
        "horizon": horizon,
        "reference_close": reference,
        "predicted_low": low,
        "expected_close": close,
        "predicted_high": high,
        "weights": weights,
        "experts": expert_predictions,
        "attribution": attribution,
    }


def run_walk_forward_simulation(
    ung: pd.DataFrame,
    *,
    outside_rth: pd.DataFrame | None = None,
    min_train_rows: int = 252,
    step: int = 5,
    max_origins: int = 252,
) -> dict[str, Any]:
    """Run a future-blind adaptive simulation and return forecasts and metrics.

    By default this simulates roughly five years of weekly forecast origins.
    Each forecast is scored only after the full target horizon has elapsed.
    """
    if step < 1 or max_origins < 1:
        raise ValueError("step and max_origins must be positive")
    ung = ung.sort_index().copy()
    if len(ung) < min_train_rows + 30:
        raise ValueError(f"need at least {min_train_rows + 30} OHLCV rows")
    features = build_features(ung, outside_rth=outside_rth)
    features.attrs["close"] = ung["close"]
    targets = {h: build_forward_targets(ung, h) for h in range(1, 6)}
    first = min_train_rows + 5
    origins = list(range(first, max(first, len(ung) - 5), step))[-max_origins:]
    pending: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    range_history: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"low": [], "high": [], "joint": []})

    for origin in origins:
        still_pending = []
        for item in pending:
            if item["mature_index"] <= origin:
                actual = item["actual"]
                range_history[item["horizon"]]["low"].append(
                    max(0.0, item["predicted_low_return"] - actual["low"])
                )
                range_history[item["horizon"]]["high"].append(
                    max(0.0, actual["high"] - item["predicted_high_return"])
                )
                range_history[item["horizon"]]["joint"].append(max(
                    max(0.0, item["predicted_low_return"] - actual["low"]),
                    max(0.0, actual["high"] - item["predicted_high_return"]),
                ))
                for factor, prediction in item["experts"].items():
                    loss = (abs(prediction["low"] - actual["low"]) + abs(prediction["close"] - actual["close"]) + abs(prediction["high"] - actual["high"])) / 3
                    history.append({"factor": factor, "horizon": item["horizon"], "loss": loss})
                outcomes.append({**item["score"], "horizon": item["horizon"], "target_session": item["target_session"]})
            else:
                still_pending.append(item)
        pending = still_pending

        for horizon in range(1, 6):
            prediction = _predict_origin(
                features, targets[horizon], origin, horizon,
                min_train_rows=min_train_rows, history=history,
            )
            if prediction is None:
                continue
            target_index = origin + horizon
            target_date = ung.index[target_index].date().isoformat()
            origin_date = ung.index[origin].date().isoformat()
            reference = prediction["reference_close"]
            actual = {
                "low": float(ung["low"].iloc[origin + 1:target_index + 1].min()) / reference - 1,
                "close": float(ung["close"].iloc[target_index]) / reference - 1,
                "high": float(ung["high"].iloc[origin + 1:target_index + 1].max()) / reference - 1,
            }
            experts = prediction["experts"]
            low_offset, high_offset = _calibration_offsets(range_history, horizon)
            calibrated_low = min(prediction["predicted_low"] - reference * low_offset, prediction["expected_close"])
            calibrated_high = max(prediction["predicted_high"] + reference * high_offset, prediction["expected_close"])
            row = {
                "origin_session": origin_date,
                "target_session": target_date,
                "horizon": horizon,
                "reference_close": reference,
                "predicted_low": calibrated_low,
                "expected_close": prediction["expected_close"],
                "predicted_high": calibrated_high,
                "factor_weights": prediction["weights"],
                "factor_contributions": prediction["attribution"],
            }
            forecasts.append(row)
            pending.append({
                "mature_index": target_index,
                "horizon": horizon,
                "target_session": target_date,
                "actual": actual,
                "experts": experts,
                # Save raw misses for calibration; scores below still evaluate
                # the calibrated range. Feeding calibrated misses back into
                # their own correction would gradually shrink the interval.
                "predicted_low_return": prediction["predicted_low"] / reference - 1,
                "predicted_high_return": prediction["predicted_high"] / reference - 1,
                "score": {
                    "origin_session": origin_date,
                    "absolute_close_error_pct": abs(prediction["expected_close"] / reference - 1 - actual["close"]) * 100,
                    "persistence_close_error_pct": abs(actual["close"]) * 100,
                    "range_contains_day": calibrated_low <= reference * (1 + actual["low"]) and calibrated_high >= reference * (1 + actual["high"]),
                    "predicted_width_pct": (calibrated_high - calibrated_low) / reference * 100,
                    "actual_width_pct": (actual["high"] - actual["low"]) * 100,
                },
            })

    # Mature all predictions whose complete target window exists, including the final origin block.
    for item in pending:
        if item["mature_index"] < len(ung):
            actual = item["actual"]
            range_history[item["horizon"]]["low"].append(
                max(0.0, item["predicted_low_return"] - actual["low"])
            )
            range_history[item["horizon"]]["high"].append(
                max(0.0, actual["high"] - item["predicted_high_return"])
            )
            range_history[item["horizon"]]["joint"].append(max(
                max(0.0, item["predicted_low_return"] - actual["low"]),
                max(0.0, actual["high"] - item["predicted_high_return"]),
            ))
            for factor, prediction in item["experts"].items():
                loss = (abs(prediction["low"] - actual["low"]) + abs(prediction["close"] - actual["close"]) + abs(prediction["high"] - actual["high"])) / 3
                history.append({"factor": factor, "horizon": item["horizon"], "loss": loss})
            outcomes.append({**item["score"], "horizon": item["horizon"], "target_session": item["target_session"]})

    outcome_frame = pd.DataFrame(outcomes)
    metrics = []
    if not outcome_frame.empty:
        for horizon, group in outcome_frame.groupby("horizon"):
            metrics.append({
                "horizon": int(horizon),
                "forecasts": int(len(group)),
                "mean_absolute_close_error_pct": float(group["absolute_close_error_pct"].mean()),
                "persistence_close_error_pct": float(group["persistence_close_error_pct"].mean()),
                "range_coverage": float(group["range_contains_day"].mean()),
                "mean_predicted_width_pct": float(group["predicted_width_pct"].mean()),
                "mean_actual_width_pct": float(group["actual_width_pct"].mean()),
            })
    current_forecasts = []
    current_origin = len(ung) - 1
    for horizon in range(1, 6):
        prediction = _predict_origin(
            features, targets[horizon], current_origin, horizon,
            min_train_rows=min_train_rows, history=history,
        )
        if prediction is None:
            continue
        low_offset, high_offset = _calibration_offsets(range_history, horizon)
        current_low = min(prediction["predicted_low"] - prediction["reference_close"] * low_offset, prediction["expected_close"])
        current_high = max(prediction["predicted_high"] + prediction["reference_close"] * high_offset, prediction["expected_close"])
        current_forecasts.append({
            "origin_session": ung.index[current_origin].date().isoformat(),
            "horizon": horizon,
            "reference_close": prediction["reference_close"],
            "predicted_low": current_low,
            "expected_close": prediction["expected_close"],
            "predicted_high": current_high,
            "factor_weights": prediction["weights"],
            "factor_contributions": prediction["attribution"],
        })
    latest_weights = current_forecasts[-1]["factor_weights"] if current_forecasts else {}
    active_factor_groups = [
        name for name, columns in FACTOR_GROUPS.items()
        if any(column in features and int(features[column].notna().sum()) >= min(100, min_train_rows) for column in columns)
    ]
    return {
        "status": "SIMULATED" if forecasts else "INSUFFICIENT_HISTORY",
        "forecast_model": "HistGradientBoostingRegressor factor-group experts",
        "range_calibration": "rolling 90th percentile of prior joint path misses",
        "factor_groups": active_factor_groups,
        "bars": len(ung),
        "simulation_origins": len(origins),
        "first_origin": ung.index[origins[0]].date().isoformat() if origins else None,
        "last_origin": ung.index[origins[-1]].date().isoformat() if origins else None,
        "metrics_by_horizon": metrics,
        "latest_factor_weights": latest_weights,
        "latest_forecasts": current_forecasts,
        "last_simulated_forecasts": forecasts[-5:],
        "forecasts": forecasts,
    }
