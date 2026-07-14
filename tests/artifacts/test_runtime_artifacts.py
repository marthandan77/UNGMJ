from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import pytest

from ung_forecast.artifacts import (
    ArtifactState,
    discover_horizon_artifacts,
    load_validated_artifacts,
)
from ung_forecast.horizons import HorizonKey
from ung_forecast.models.calibration import MulticlassProbabilityCalibrator
from ung_forecast.models.elastic_net import ElasticNetMultinomialModel
from ung_forecast.validation.metrics import CLASS_ORDER


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fitted_objects() -> tuple[ElasticNetMultinomialModel, MulticlassProbabilityCalibrator]:
    rng = np.random.default_rng(7)
    features = pd.DataFrame(rng.normal(size=(180, 2)), columns=["a", "b"])
    target = pd.Series(CLASS_ORDER * 60)
    model = ElasticNetMultinomialModel()
    model.fit(features, target)
    raw_rows = []
    for probability in model.predict_probabilities(features):
        raw_rows.append(
            {
                "LOWER_FIRST": probability.lower_first,
                "UPPER_FIRST": probability.upper_first,
                "NEITHER": probability.neither,
            }
        )
    raw = pd.DataFrame(raw_rows, index=features.index, columns=CLASS_ORDER)
    calibrator = MulticlassProbabilityCalibrator()
    calibrator.fit(raw, target)
    return model, calibrator


def write_artifacts(tmp_path, *, model_object=None, configuration_hash="runtime-config"):
    directory = tmp_path / HorizonKey.MINUTES_60.value
    directory.mkdir(parents=True)
    model, calibrator = fitted_objects()
    joblib.dump(model if model_object is None else model_object, directory / "model.joblib")
    joblib.dump(calibrator, directory / "calibrator.joblib")
    manifest = {
        "horizon": "60m",
        "created_at": datetime(2026, 7, 12, tzinfo=UTC).isoformat(),
        "model_version": "model-v1",
        "feature_version": "features-v1",
        "data_version": "data-v1",
        "configuration_hash": configuration_hash,
        "feature_names": ["a", "b"],
        "includes_comparison": False,
        "statistical_approved": True,
        "approval_reasons": [],
        "metrics": {
            "brier_score": 0.5,
            "log_loss": 0.9,
            "calibration_error": 0.05,
            "sample_count": 500,
            "baseline_brier_score": 0.7,
            "baseline_log_loss": 1.1,
        },
        "model_file": {
            "relative_path": "model.joblib",
            "sha256": sha256(directory / "model.joblib"),
        },
        "calibrator_file": {
            "relative_path": "calibrator.joblib",
            "sha256": sha256(directory / "calibrator.joblib"),
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_validated_objects_are_type_checked_after_checksum_verification(tmp_path) -> None:
    write_artifacts(tmp_path)
    result = discover_horizon_artifacts(
        tmp_path,
        expected_configuration_hash="runtime-config",
    )[HorizonKey.MINUTES_60]
    loaded = load_validated_artifacts(result)
    assert result.state is ArtifactState.VALIDATED
    assert loaded.model.feature_names == ("a", "b")
    assert isinstance(loaded.calibrator, MulticlassProbabilityCalibrator)


def test_configuration_mismatch_is_invalid(tmp_path) -> None:
    write_artifacts(tmp_path)
    result = discover_horizon_artifacts(
        tmp_path,
        expected_configuration_hash="different-config",
    )[HorizonKey.MINUTES_60]
    assert result.state is ArtifactState.INVALID
    assert "configuration hash" in result.errors[0]


def test_checksum_valid_but_wrong_runtime_type_is_rejected(tmp_path) -> None:
    write_artifacts(tmp_path, model_object={"not": "a model"})
    result = discover_horizon_artifacts(
        tmp_path,
        expected_configuration_hash="runtime-config",
    )[HorizonKey.MINUTES_60]
    assert result.state is ArtifactState.VALIDATED
    with pytest.raises(TypeError, match="unexpected runtime type"):
        load_validated_artifacts(result)
