from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from ung_forecast.artifacts import discover_horizon_artifacts
from ung_forecast.data.runtime_loader import RuntimeDataSet
from ung_forecast.data.schemas import DataProvenance, MarketDataBundle
from ung_forecast.features import build_feature_frame
from ung_forecast.horizons import HorizonKey
from ung_forecast.models.calibration import MulticlassProbabilityCalibrator
from ung_forecast.models.elastic_net import ElasticNetMultinomialModel
from ung_forecast.runtime_forecast import generate_runtime_probability_forecast
from ung_forecast.validation.metrics import CLASS_ORDER


def market_frame() -> pd.DataFrame:
    timestamps: list[pd.Timestamp] = []
    for day in pd.bdate_range("2026-03-02", periods=40):
        start = pd.Timestamp(day.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        timestamps.extend(pd.date_range(start, periods=12, freq="5min"))
    index = pd.DatetimeIndex(timestamps)
    step = np.arange(len(index), dtype=float)
    close = 10.0 + 0.002 * step + 0.08 * np.sin(step / 6.0)
    return pd.DataFrame(
        {
            "Open": close - 0.01,
            "High": close + 0.03,
            "Low": close - 0.03,
            "Close": close,
            "Volume": 1000 + (step % 12) * 10 + step % 5,
        },
        index=index,
    )


def probability_frame(model, features: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "LOWER_FIRST": item.lower_first,
                "UPPER_FIRST": item.upper_first,
                "NEITHER": item.neither,
            }
            for item in model.predict_probabilities(features)
        ],
        index=features.index,
        columns=CLASS_ORDER,
    )


def test_verified_artifact_generates_calibrated_research_probability(tmp_path) -> None:
    frame = market_frame()
    built = build_feature_frame(frame, horizon=HorizonKey.MINUTES_60)
    features = built.frame.dropna()
    target = pd.Series(
        np.resize(np.array(CLASS_ORDER), len(features)),
        index=features.index,
    )
    model = ElasticNetMultinomialModel()
    model.fit(features, target)
    raw = probability_frame(model, features)
    calibrator = MulticlassProbabilityCalibrator()
    calibrator.fit(raw, target)

    directory = tmp_path / "60m"
    directory.mkdir()
    model_path = directory / "model.joblib"
    calibrator_path = directory / "calibrator.joblib"
    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "horizon": "60m",
        "created_at": datetime(2026, 7, 12, tzinfo=UTC).isoformat(),
        "model_version": "model-v1",
        "feature_version": "features-v1",
        "data_version": "data-v1",
        "configuration_hash": "runtime-config",
        "feature_names": list(features.columns),
        "includes_comparison": False,
        "statistical_approved": True,
        "approval_reasons": [],
        "metrics": {
            "brier_score": 0.5,
            "log_loss": 0.9,
            "calibration_error": 0.05,
            "sample_count": len(features),
            "baseline_brier_score": 0.7,
            "baseline_log_loss": 1.1,
        },
        "model_file": {"relative_path": model_path.name, "sha256": digest(model_path)},
        "calibrator_file": {
            "relative_path": calibrator_path.name,
            "sha256": digest(calibrator_path),
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    artifact = discover_horizon_artifacts(
        tmp_path,
        expected_configuration_hash="runtime-config",
    )[HorizonKey.MINUTES_60]
    bundle = MarketDataBundle(
        frame=frame,
        provenance=DataProvenance(
            provider="test",
            symbol="UNG",
            interval="5m",
            downloaded_at=datetime(2026, 7, 12, tzinfo=UTC),
            source_timezone="America/New_York",
            normalized_timezone="America/New_York",
            adjusted_prices=False,
        ),
    )
    runtime = RuntimeDataSet(
        bundles_by_interval={"5m": bundle},
        source_by_interval={"5m": "live"},
        errors_by_interval={},
    )

    result = generate_runtime_probability_forecast(artifact, runtime)

    total = (
        result.probabilities.lower_first
        + result.probabilities.upper_first
        + result.probabilities.neither
    )
    assert total == 1.0
    assert result.timestamp == features.index[-1]
    assert result.data_source == "live"
