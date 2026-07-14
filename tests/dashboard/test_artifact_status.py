from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ung_forecast.artifacts import (
    ArtifactFile,
    ArtifactLoadResult,
    ArtifactState,
    HorizonArtifactManifest,
    StatisticalMetricsSnapshot,
)
from ung_forecast.dashboard import build_artifact_status_rows
from ung_forecast.horizons import HORIZON_SPECS, HorizonKey


def manifest(horizon: HorizonKey, *, approved: bool) -> HorizonArtifactManifest:
    return HorizonArtifactManifest(
        horizon=horizon,
        created_at=datetime(2026, 7, 12, tzinfo=UTC),
        model_version="model-v1",
        feature_version="features-v1",
        data_version="data-v1",
        configuration_hash="12345678abcdef",
        feature_names=("return_1", "volatility_12"),
        statistical_approved=approved,
        approval_reasons=() if approved else ("insufficient_samples",),
        metrics=StatisticalMetricsSnapshot(
            brier_score=0.60,
            log_loss=0.95,
            calibration_error=0.07,
            sample_count=500,
            baseline_brier_score=0.70,
            baseline_log_loss=1.10,
        ),
        model_file=ArtifactFile(relative_path="model.joblib", sha256="a" * 64),
        calibrator_file=ArtifactFile(relative_path="calibrator.joblib", sha256="b" * 64),
    )


def test_status_rows_preserve_canonical_horizon_order() -> None:
    results: dict[HorizonKey, ArtifactLoadResult] = {}
    for horizon in HORIZON_SPECS:
        results[horizon] = ArtifactLoadResult(
            horizon=horizon,
            state=ArtifactState.UNAVAILABLE,
            manifest=None,
            artifact_directory=None,
            errors=("manifest_not_found",),
        )
    results[HorizonKey.MINUTES_60] = ArtifactLoadResult(
        horizon=HorizonKey.MINUTES_60,
        state=ArtifactState.VALIDATED,
        manifest=manifest(HorizonKey.MINUTES_60, approved=True),
        artifact_directory=Path("artifacts/models/60m"),
        errors=(),
    )

    rows = build_artifact_status_rows(results)
    assert [row.horizon for row in rows] == [
        specification.display_name for specification in HORIZON_SPECS.values()
    ]
    assert rows[0].state == "VALIDATED"
    assert rows[0].samples == 500
    assert rows[1].state == "UNAVAILABLE"
    assert rows[1].detail == "manifest_not_found"


def test_research_only_row_displays_approval_reason() -> None:
    results = {
        horizon: ArtifactLoadResult(
            horizon=horizon,
            state=ArtifactState.RESEARCH_ONLY,
            manifest=manifest(horizon, approved=False),
            artifact_directory=Path("artifacts/models") / horizon.value,
            errors=(),
        )
        for horizon in HORIZON_SPECS
    }
    rows = build_artifact_status_rows(results)
    assert all(row.detail == "insufficient_samples" for row in rows)
