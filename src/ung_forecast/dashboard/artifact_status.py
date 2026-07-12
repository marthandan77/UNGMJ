"""Convert artifact discovery results into Streamlit-ready status rows."""

from __future__ import annotations

from dataclasses import dataclass

from ung_forecast.artifacts import ArtifactLoadResult
from ung_forecast.horizons import HORIZON_SPECS, HorizonKey


@dataclass(frozen=True, slots=True)
class ArtifactStatusRow:
    horizon: str
    state: str
    model_version: str
    feature_version: str
    samples: int | None
    brier_score: float | None
    calibration_error: float | None
    detail: str


def build_artifact_status_rows(
    results: dict[HorizonKey, ArtifactLoadResult],
) -> tuple[ArtifactStatusRow, ...]:
    rows: list[ArtifactStatusRow] = []
    for horizon in HORIZON_SPECS:
        result = results[horizon]
        manifest = result.manifest
        detail = (
            "; ".join(manifest.approval_reasons)
            if manifest is not None and manifest.approval_reasons
            else "; ".join(result.errors)
        )
        rows.append(
            ArtifactStatusRow(
                horizon=HORIZON_SPECS[horizon].display_name,
                state=result.state.value,
                model_version=(manifest.model_version if manifest is not None else "—"),
                feature_version=(manifest.feature_version if manifest is not None else "—"),
                samples=(manifest.metrics.sample_count if manifest is not None else None),
                brier_score=(manifest.metrics.brier_score if manifest is not None else None),
                calibration_error=(
                    manifest.metrics.calibration_error if manifest is not None else None
                ),
                detail=detail or "Ready",
            )
        )
    return tuple(rows)
