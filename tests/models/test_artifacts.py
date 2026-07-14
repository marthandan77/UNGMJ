from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ung_forecast.horizons import HorizonKey
from ung_forecast.models.artifacts import ModelArtifactMetadata, ModelArtifactStore


def metadata() -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        horizon=HorizonKey.MINUTES_60,
        model_version="model-v1",
        feature_version="features-v1",
        data_version="data-v1",
        configuration_hash="12345678abcdef",
        created_at=datetime(2026, 7, 11, 12, 0, tzinfo=UTC),
        approved=False,
        includes_comparison=True,
        validation_metrics={"brier_score": 0.55, "sample_count": 300},
    )


def test_model_artifact_round_trip_and_checksum(tmp_path) -> None:
    store = ModelArtifactStore(tmp_path)
    finalized = store.write({"payload": [1, 2, 3]}, metadata())
    loaded, loaded_metadata = store.load(HorizonKey.MINUTES_60, "model-v1")

    assert loaded == {"payload": [1, 2, 3]}
    assert loaded_metadata.artifact_sha256 == finalized.artifact_sha256
    assert len(finalized.artifact_sha256) == 64


def test_model_artifact_refuses_overwrite(tmp_path) -> None:
    store = ModelArtifactStore(tmp_path)
    store.write({"payload": 1}, metadata())
    with pytest.raises(FileExistsError):
        store.write({"payload": 2}, metadata())


def test_model_artifact_detects_tampering(tmp_path) -> None:
    store = ModelArtifactStore(tmp_path)
    store.write({"payload": 1}, metadata())
    artifact = tmp_path / "60m" / "model-v1" / "model.joblib"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.load(HorizonKey.MINUTES_60, "model-v1")
