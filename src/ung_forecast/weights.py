"""Dynamic model averaging for HistGradientBoosting feature-group experts."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


DMA_FORGETTING_FACTOR = 0.98
DMA_LEARNING_RATE = 0.5
DMA_MIN_SAMPLES = 20


def _outcome_loss(row: dict[str, Any]) -> float:
    """Scaled low/close/high absolute error used as a proper range-oriented proxy."""
    scale = max(float(row.get("reference_close", 1.0) or 1.0), 1e-6)
    errors = [abs(float(row[key])) / scale for key in ("low_error", "close_error", "high_error") if key in row]
    return sum(errors) / len(errors) if errors else abs(float(row.get("close_error", 0.0))) / scale


def compute_model_weights(
    outcomes: list[dict[str, Any]], *, min_samples: int = DMA_MIN_SAMPLES,
    forgetting_factor: float = DMA_FORGETTING_FACTOR,
    learning_rate: float = DMA_LEARNING_RATE,
) -> list[dict[str, Any]]:
    """Replay matured, horizon-specific forecast losses into DMA expert weights.

    All component forecasts use HistGradientBoosting; the experts differ by the
    feature group they see. Until there is enough evidence, they retain equal
    prior weights. The consensus itself never trains its own weights.
    """
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        model_name = str(row.get("model_name", ""))
        if not (model_name == "hist_gradient_boosting" or model_name.startswith("hgb_expert_")):
            continue
        grouped[int(row["horizon"])].append(row)

    result: list[dict[str, Any]] = []
    for horizon, rows in sorted(grouped.items()):
        by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_name[str(row["model_name"])].append(row)
        # Repeated scans can create multiple forecast vintages for the same
        # target session. Learn once per target/model using the latest scored row.
        for name, model_rows in list(by_name.items()):
            unique: dict[str, dict[str, Any]] = {}
            for row in sorted(model_rows, key=lambda item: str(item.get("scored_at", ""))):
                key = str(row.get("target_session", row.get("scored_at", len(unique))))
                unique[key] = row
            by_name[name] = list(unique.values())
        names = sorted(by_name)
        if not names:
            continue
        eligible = [name for name in names if len(by_name[name]) >= min_samples]
        active = eligible or names
        weights = {name: 1.0 / len(active) for name in active}

        if eligible:
            by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for name in eligible:
                for row in by_name[name]:
                    by_date[str(row.get("target_session", row.get("scored_at", "")))].append(row)
            for date in sorted(by_date):
                current = [row for row in by_date[date] if str(row["model_name"]) in weights]
                if len(current) < 2:
                    # Do not penalize a new expert for dates before it existed.
                    continue
                losses = {str(row["model_name"]): _outcome_loss(row) for row in current}
                scale = max(sum(losses.values()) / len(losses), 1e-5)
                updated = {
                    name: weights[name] ** forgetting_factor
                    * math.exp(-learning_rate * min(losses.get(name, scale) / scale, 8.0))
                    for name in weights
                }
                total = sum(updated.values())
                weights = {name: value / total for name, value in updated.items()}

        # Reserve a small amount of probability for each expert so no model is
        # permanently discarded after one regime-specific run of poor results.
        floor = min(0.02, 0.5 / len(weights))
        adjusted = {name: max(floor, value) for name, value in weights.items()}
        total = sum(adjusted.values())
        weights = {name: value / total for name, value in adjusted.items()}
        for name in names:
            recent_rows = by_name[name][-60:]
            learned = name in eligible
            result.append({
                "model_name": name,
                "horizon": horizon,
                "sample_count": len(by_name[name]),
                "recent_sample_count": len(recent_rows),
                "mean_abs_range_error_pct": 100 * sum(_outcome_loss(row) for row in recent_rows) / max(len(recent_rows), 1),
                "weight": weights.get(name, 0.0),
                "status": "DMA" if learned else "WARMING_UP",
                "algorithm": "HistGradientBoostingRegressor",
            })
    return result


def build_dma_consensus(
    forecasts: list[dict[str, Any]], weights: list[dict[str, Any]], *, minimum_models: int = 2,
) -> list[dict[str, Any]]:
    """Blend HGB feature-group experts with dynamic horizon-specific weights."""
    weight_by_horizon: dict[int, dict[str, float]] = defaultdict(dict)
    learned_count_by_horizon: dict[int, int] = defaultdict(int)
    samples_by_horizon: dict[int, int] = {}
    for row in weights:
        horizon = int(row["horizon"])
        weight_by_horizon[horizon][str(row["model_name"])] = float(row["weight"])
        if row.get("status") == "DMA":
            learned_count_by_horizon[horizon] += 1
        samples_by_horizon[horizon] = min(
            samples_by_horizon.get(horizon, int(row.get("sample_count", 0))),
            int(row.get("sample_count", 0)),
        )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for forecast in forecasts:
        if forecast.get("model_name") != "dma_consensus":
            grouped[int(forecast["horizon"])].append(forecast)

    result = []
    fields = ("predicted_low", "expected_close", "predicted_high", "probability_up", "uncertainty")
    for horizon, candidates in sorted(grouped.items()):
        if len(candidates) < minimum_models:
            continue
        learned = weight_by_horizon.get(horizon, {})
        raw = {row["model_name"]: learned.get(row["model_name"], 0.0) for row in candidates}
        total = sum(raw.values())
        if total <= 0:
            raw = {row["model_name"]: 1.0 for row in candidates}
            total = float(len(candidates))
        normalized = {name: value / total for name, value in raw.items()}
        consensus = {
            key: sum(float(row[key]) * normalized[row["model_name"]] for row in candidates)
            for key in fields
        }
        consensus["predicted_low"] = min(consensus["predicted_low"], consensus["expected_close"])
        consensus["predicted_high"] = max(consensus["predicted_high"], consensus["expected_close"])
        base = candidates[0]
        result.append({
            "origin_session": base["origin_session"],
            "target_session": base["target_session"],
            "horizon": horizon,
            "model_name": "dma_consensus",
            "algorithm": "HistGradientBoostingRegressor + Dynamic Model Averaging",
            "reference_close": base["reference_close"],
            **consensus,
            "status": "PROVISIONAL",
            "learning_status": "DMA" if learned_count_by_horizon.get(horizon, 0) >= 2 else "WARMING_UP",
            "learning_sample_count": samples_by_horizon.get(horizon, 0),
            "component_weights": normalized,
            "component_models": sorted(normalized),
        })
    return result
