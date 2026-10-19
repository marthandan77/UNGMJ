"""Slow, bounded model-weight learning from matured out-of-sample outcomes."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def compute_model_weights(
    outcomes: list[dict[str, Any]], *, min_samples: int = 20,
    floor: float = 0.05, ceiling: float = 0.50,
) -> list[dict[str, Any]]:
    """Learn recent per-horizon model weights from matured out-of-sample errors."""
    groups: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in outcomes:
        # The adaptive consensus is an output of this learner, not a base model.
        if row.get("model_name") == "adaptive_consensus":
            continue
        groups[int(row["horizon"])][str(row["model_name"])].append(abs(float(row["close_error"])))
    result: list[dict[str, Any]] = []
    for horizon, models in sorted(groups.items()):
        eligible = {name: errors for name, errors in models.items() if len(errors) >= min_samples}
        if not eligible:
            continue
        # Recent outcomes let the system adapt when relative model performance
        # changes, while the sample threshold prevents reaction to tiny samples.
        recent = {name: errors[-60:] for name, errors in eligible.items()}
        raw = {
            name: 1.0 / max(sum(errors) / len(errors), 1e-9)
            for name, errors in recent.items()
        }
        total = sum(raw.values())
        weights = {name: value / total for name, value in raw.items()}
        for name, weight in weights.items():
            result.append({
                "model_name": name,
                "horizon": horizon,
                "sample_count": len(recent[name]),
                "mean_abs_close_error": 1.0 / raw[name],
                "weight": min(ceiling, max(floor, weight)),
                "status": "LEARNED",
            })
    return result


def build_adaptive_consensus(
    forecasts: list[dict[str, Any]], weights: list[dict[str, Any]], *, minimum_models: int = 2,
) -> list[dict[str, Any]]:
    """Create a separately labeled, weighted forecast when enough models have evidence."""
    weight_by_horizon: dict[int, dict[str, float]] = defaultdict(dict)
    for row in weights:
        weight_by_horizon[int(row["horizon"])][str(row["model_name"])] = float(row["weight"])

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for forecast in forecasts:
        if forecast.get("model_name") != "adaptive_consensus":
            grouped[int(forecast["horizon"])].append(forecast)

    result = []
    fields = (
        "predicted_low", "expected_close", "predicted_high", "probability_up", "uncertainty",
    )
    for horizon, candidates in sorted(grouped.items()):
        learned = weight_by_horizon.get(horizon, {})
        eligible = [row for row in candidates if row["model_name"] in learned]
        if len(eligible) < minimum_models:
            continue
        total = sum(learned[row["model_name"]] for row in eligible)
        if total <= 0:
            continue
        normalized = {row["model_name"]: learned[row["model_name"]] / total for row in eligible}
        consensus = {key: sum(float(row[key]) * normalized[row["model_name"]] for row in eligible) for key in fields}
        base = eligible[0]
        result.append({
            "origin_session": base["origin_session"],
            "target_session": base["target_session"],
            "horizon": horizon,
            "model_name": "adaptive_consensus",
            "reference_close": base["reference_close"],
            **consensus,
            "status": "PROVISIONAL",
            "learning_status": "ADAPTIVE",
            "learning_sample_count": min(int(next(
                row["sample_count"] for row in weights
                if int(row["horizon"]) == horizon and row["model_name"] == name
            )) for name in normalized),
            "component_weights": normalized,
            "component_models": sorted(normalized),
        })
    return result
