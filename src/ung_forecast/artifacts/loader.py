"""Discover model artifacts and reject incomplete or tampered bundles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from ung_forecast.horizons import HORIZON_SPECS, HorizonKey

from .manifest import ArtifactFile, HorizonArtifactManifest


class ArtifactState(StrEnum):
    VALIDATED = "VALIDATED"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class ArtifactLoadResult:
    horizon: HorizonKey
    state: ArtifactState
    manifest: HorizonArtifactManifest | None
    artifact_directory: Path | None
    errors: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_checked(root: Path, artifact: ArtifactFile) -> Path:
    root_resolved = root.resolve()
    candidate = (root / artifact.relative_path).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError("Artifact path escapes its manifest directory")
    if not candidate.is_file():
        raise FileNotFoundError(f"Artifact file is missing: {artifact.relative_path}")
    observed = _sha256(candidate)
    if observed != artifact.sha256:
        raise ValueError(f"Checksum mismatch: {artifact.relative_path}")
    return candidate


def load_manifest(path: str | Path) -> HorizonArtifactManifest:
    manifest_path = Path(path)
    try:
        return HorizonArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        raise
    except (OSError, ValidationError, ValueError) as exc:
        raise ValueError(f"Invalid artifact manifest: {manifest_path}") from exc


def _discover_one(root: Path, horizon: HorizonKey) -> ArtifactLoadResult:
    artifact_directory = root / horizon.value
    manifest_path = artifact_directory / "manifest.json"
    if not manifest_path.exists():
        return ArtifactLoadResult(
            horizon=horizon,
            state=ArtifactState.UNAVAILABLE,
            manifest=None,
            artifact_directory=None,
            errors=("manifest_not_found",),
        )
    try:
        manifest = load_manifest(manifest_path)
        if manifest.horizon is not horizon:
            raise ValueError(
                f"Manifest horizon {manifest.horizon.value} does not match directory {horizon.value}"
            )
        _resolve_checked(artifact_directory, manifest.model_file)
        _resolve_checked(artifact_directory, manifest.calibrator_file)
    except (OSError, ValueError) as exc:
        return ArtifactLoadResult(
            horizon=horizon,
            state=ArtifactState.INVALID,
            manifest=None,
            artifact_directory=artifact_directory,
            errors=(str(exc),),
        )

    state = (
        ArtifactState.VALIDATED
        if manifest.statistical_approved
        else ArtifactState.RESEARCH_ONLY
    )
    return ArtifactLoadResult(
        horizon=horizon,
        state=state,
        manifest=manifest,
        artifact_directory=artifact_directory,
        errors=(),
    )


def discover_horizon_artifacts(root: str | Path) -> dict[HorizonKey, ArtifactLoadResult]:
    artifact_root = Path(root)
    return {
        horizon: _discover_one(artifact_root, horizon)
        for horizon in HORIZON_SPECS
    }
