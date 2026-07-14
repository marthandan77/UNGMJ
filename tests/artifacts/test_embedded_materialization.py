from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from ung_forecast.artifacts import ArtifactState, discover_horizon_artifacts
from ung_forecast.horizons import HorizonKey


def _manifest(model_sha: str, calibrator_sha: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "horizon": "60m",
        "created_at": "2026-01-01T00:00:00Z",
        "model_version": "test-v1",
        "feature_version": "features-v1",
        "data_version": "test-data",
        "configuration_hash": "config-hash",
        "feature_names": ["return_3"],
        "includes_comparison": False,
        "probability_mode": "raw",
        "elastic_net_c": 0.1,
        "elastic_net_l1_ratio": 1.0,
        "statistical_approved": True,
        "approval_reasons": [],
        "metrics": {
            "brier_score": 0.5,
            "log_loss": 0.8,
            "calibration_error": 0.02,
            "sample_count": 300,
            "baseline_brier_score": 0.6,
            "baseline_log_loss": 0.9,
        },
        "model_file": {"relative_path": "test-v1/model.joblib", "sha256": model_sha},
        "calibrator_file": {
            "relative_path": "test-v1/calibrator.joblib",
            "sha256": calibrator_sha,
        },
    }


def _write_bundle(root: Path, *, corrupt_model: bool = False) -> tuple[bytes, bytes]:
    horizon = root / "60m"
    version = horizon / "test-v1"
    version.mkdir(parents=True)
    model = b"approved-model-payload"
    calibrator = b"approved-calibrator-payload"
    model_sha = hashlib.sha256(model).hexdigest()
    calibrator_sha = hashlib.sha256(calibrator).hexdigest()
    (horizon / "manifest.json").write_text(
        json.dumps(_manifest(model_sha, calibrator_sha)), encoding="utf-8"
    )
    model_payload = b"corrupt" if corrupt_model else model
    (version / "model.joblib.b64").write_text(
        base64.b64encode(model_payload).decode("ascii"), encoding="ascii"
    )
    (version / "calibrator.joblib.b64").write_text(
        base64.b64encode(calibrator).decode("ascii"), encoding="ascii"
    )
    return model, calibrator


def test_discovery_materializes_checksum_verified_embedded_payloads(tmp_path: Path) -> None:
    model, calibrator = _write_bundle(tmp_path)

    result = discover_horizon_artifacts(
        tmp_path,
        expected_configuration_hash="config-hash",
    )[HorizonKey.MINUTES_60]

    assert result.state is ArtifactState.VALIDATED
    assert (tmp_path / "60m/test-v1/model.joblib").read_bytes() == model
    assert (tmp_path / "60m/test-v1/calibrator.joblib").read_bytes() == calibrator


def test_discovery_rejects_embedded_payload_with_wrong_checksum(tmp_path: Path) -> None:
    _write_bundle(tmp_path, corrupt_model=True)

    result = discover_horizon_artifacts(
        tmp_path,
        expected_configuration_hash="config-hash",
    )[HorizonKey.MINUTES_60]

    assert result.state is ArtifactState.INVALID
    assert "Embedded artifact checksum mismatch" in result.errors[0]
