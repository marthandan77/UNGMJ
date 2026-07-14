"""Generate calibrated research probabilities from verified runtime inputs.

This module deliberately stops before expected-value advice. Statistical model
approval alone is not trading approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import pandas as pd

from ung_forecast.artifacts import ArtifactLoadResult, load_validated_artifacts
from ung_forecast.data import RuntimeDataSet
from ung_forecast.features import build_feature_frame
from ung_forecast.horizons import HORIZON_SPECS, HorizonKey
from ung_forecast.schemas import ProbabilityForecast
from ung_forecast.validation.metrics import CLASS_ORDER


@dataclass(frozen=True, slots=True)
class RuntimeProbabilityForecast:
    horizon: HorizonKey
    timestamp: pd.Timestamp
    current_price: float
    probabilities: ProbabilityForecast
    model_version: str
    feature_version: str
    data_version: str
    data_source: str


def generate_runtime_probability_forecast(
    artifact_result: ArtifactLoadResult,
    runtime_data: RuntimeDataSet,
) -> RuntimeProbabilityForecast:
    loaded = load_validated_artifacts(artifact_result)
    manifest = loaded.manifest
    if manifest.includes_comparison:
        raise ValueError("Comparison-dependent runtime inference is not configured")

    bundle = runtime_data.bundle_for_horizon(manifest.horizon)
    feature_result = build_feature_frame(
        bundle.frame,
        horizon=manifest.horizon,
        comparison=None,
        feature_version=manifest.feature_version,
    )
    if feature_result.feature_version != manifest.feature_version:
        raise ValueError("Runtime feature version does not match artifact manifest")
    missing = set(manifest.feature_names).difference(feature_result.frame.columns)
    if missing:
        raise ValueError(f"Runtime feature frame is missing columns: {sorted(missing)}")

    ordered = feature_result.frame.loc[:, list(manifest.feature_names)].dropna()
    if ordered.empty:
        raise ValueError("Insufficient completed history for the artifact feature schema")
    latest = ordered.iloc[[-1]]
    if tuple(latest.columns) != manifest.feature_names:
        raise ValueError("Runtime feature order does not match artifact manifest")

    raw = loaded.model.predict_probabilities(latest)[0]
    raw_frame = pd.DataFrame(
        [
            {
                "LOWER_FIRST": raw.lower_first,
                "UPPER_FIRST": raw.upper_first,
                "NEITHER": raw.neither,
            }
        ],
        index=latest.index,
        columns=CLASS_ORDER,
    )
    calibrated = loaded.calibrator.transform(raw_frame).iloc[0]
    probabilities = ProbabilityForecast(
        lower_first=float(calibrated["LOWER_FIRST"]),
        upper_first=float(calibrated["UPPER_FIRST"]),
        neither=float(calibrated["NEITHER"]),
    )
    interval = HORIZON_SPECS[manifest.horizon].source_interval
    timestamp = pd.Timestamp(latest.index[-1])
    close_value = bundle.frame["Close"].at[timestamp]
    if isinstance(close_value, bool) or not isinstance(close_value, Real):
        raise TypeError("Latest close is not a real numeric scalar")

    return RuntimeProbabilityForecast(
        horizon=manifest.horizon,
        timestamp=timestamp,
        current_price=float(close_value),
        probabilities=probabilities,
        model_version=manifest.model_version,
        feature_version=manifest.feature_version,
        data_version=manifest.data_version,
        data_source=runtime_data.source_by_interval[interval],
    )
