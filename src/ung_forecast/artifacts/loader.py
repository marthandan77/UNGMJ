"""Discover model artifacts and reject incomplete, incompatible, or tampered bundles."""

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


def resolve_checked(root: Path, artifact: ArtifactFile) -> Path:
    root_resolved = root.resolve()
    candidate = (root / artifact.relative_path).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError("Artifact path escapes its manifest directory")
    if not candidate.is_file():
        raise FileNotFoundError(f"Artifact file is missing: {artifact.relative_path}")
    if _sha256(candidate) != artifact.sha256:
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


def _discover_one(
    root: Path,
    horizon: HorizonKey,
    *,
    expected_configuration_hash: str | None,
    comparison_available: bool,
) -> ArtifactLoadResult:
    artifact_directory = root / horizon.value
    manifest_path = artifact_directory / "manifest.json"
    if not manifest_path.exists():
        return ArtifactLoadResult(horizon, ArtifactState.UNAVAILABLE, None, None, ("manifest_not_found",))
    try:
        manifest = load_manifest(manifest_path)
        if manifest.horizon is not horizon:
            raise ValueError("Manifest horizon does not match artifact directory")
        if expected_configuration_hash is not None and (
            manifest.configuration_hash != expected_configuration_hash
        ):
            raise ValueError("Artifact configuration hash does not match running application")
        if manifest.includes_comparison and not comparison_available:
            raise ValueError("Artifact requires comparison data that is unavailable")
        resolve_checked(artifact_directory, manifest.model_file)
        resolve_checked(artifact_directory, manifest.calibrator_file)
    except (OSError, ValueError) as exc:
        return ArtifactLoadResult(
            horizon, ArtifactState.INVALID, None, artifact_directory, (str(exc),)
        )

    state = ArtifactState.VALIDATED if manifest.statistical_approved else ArtifactState.RESEARCH_ONLY
    return ArtifactLoadResult(horizon, state, manifest, artifact_directory, ())


def discover_horizon_artifacts(
    root: str | Path,
    *,
    expected_configuration_hash: str | None = None,
    comparison_available: bool = False,
) -> dict[HorizonKey, ArtifactLoadResult]:
    artifact_root = Path(root)
    return {
        horizon: _discover_one(
            artifact_root,
            horizon,
            expected_configuration_hash=expected_configuration_hash,
            comparison_available=comparison_available,
        )
        for horizon in HORIZON_SPECS
    }
