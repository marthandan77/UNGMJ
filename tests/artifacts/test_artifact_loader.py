from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ung_forecast.artifacts import ArtifactState, discover_horizon_artifacts
from ung_forecast.horizons import HorizonKey


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_bundle(
    root: Path,
    horizon: HorizonKey,
    *,
    approved: bool,
    corrupt_model: bool = False,
) -> None:
    directory = root / horizon.value
    directory.mkdir(parents=True)
    model_payload = b"model-v1"
    calibrator_payload = b"calibrator-v1"
    (directory / "model.joblib").write_bytes(
        b"tampered" if corrupt_model else model_payload
    )
    (directory / "calibrator.joblib").write_bytes(calibrator_payload)
    manifest = {
        "schema_version": "1",
        "horizon": horizon.value,
        "created_at": datetime(2026, 7, 12, tzinfo=UTC).isoformat(),
        "model_version": "model-v1",
        "feature_version": "features-v1",
        "data_version": "data-v1",
        "configuration_hash": "12345678abcdef",
        "feature_names": ["return_1", "volatility_12"],
        "statistical_approved": approved,
        "approval_reasons": [] if approved else ["brier_not_better_than_baseline"],
        "metrics": {
            "brier_score": 0.60,
            "log_loss": 0.95,
            "calibration_error": 0.07,
            "sample_count": 500,
            "baseline_brier_score": 0.70,
            "baseline_log_loss": 1.10,
        },
        "model_file": {
            "relative_path": "model.joblib",
            "sha256": digest(model_payload),
        },
        "calibrator_file": {
            "relative_path": "calibrator.joblib",
            "sha256": digest(calibrator_payload),
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_discovery_marks_missing_horizons_unavailable(tmp_path) -> None:
    results = discover_horizon_artifacts(tmp_path)
    assert results[HorizonKey.MINUTES_60].state is ArtifactState.UNAVAILABLE
    assert results[HorizonKey.MINUTES_60].errors == ("manifest_not_found",)


def test_approved_bundle_with_valid_checksums_is_validated(tmp_path) -> None:
    write_bundle(tmp_path, HorizonKey.MINUTES_60, approved=True)
    result = discover_horizon_artifacts(tmp_path)[HorizonKey.MINUTES_60]
    assert result.state is ArtifactState.VALIDATED
    assert result.manifest is not None
    assert result.manifest.model_version == "model-v1"


def test_failed_statistical_bundle_is_research_only(tmp_path) -> None:
    write_bundle(tmp_path, HorizonKey.HOURS_4, approved=False)
    result = discover_horizon_artifacts(tmp_path)[HorizonKey.HOURS_4]
    assert result.state is ArtifactState.RESEARCH_ONLY
    assert result.manifest is not None
    assert result.manifest.approval_reasons == (
        "brier_not_better_than_baseline",
    )


def test_checksum_mismatch_is_invalid(tmp_path) -> None:
    write_bundle(
        tmp_path,
        HorizonKey.DAY_1,
        approved=True,
        corrupt_model=True,
    )
    result = discover_horizon_artifacts(tmp_path)[HorizonKey.DAY_1]
    assert result.state is ArtifactState.INVALID
    assert result.manifest is None
    assert "Checksum mismatch" in result.errors[0]


def test_manifest_cannot_escape_artifact_directory(tmp_path) -> None:
    write_bundle(tmp_path, HorizonKey.DAYS_2, approved=True)
    manifest_path = tmp_path / HorizonKey.DAYS_2.value / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["model_file"]["relative_path"] = "../model.joblib"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = discover_horizon_artifacts(tmp_path)[HorizonKey.DAYS_2]
    assert result.state is ArtifactState.INVALID
