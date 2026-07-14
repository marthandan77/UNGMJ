"""Materialize repository-embedded model payloads with checksum verification."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .loader import load_manifest
from .manifest import ArtifactFile


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialize_file(horizon_root: Path, artifact: ArtifactFile) -> Path:
    target = horizon_root / artifact.relative_path
    if target.is_file() and _sha256_file(target) == artifact.sha256:
        return target

    embedded = target.with_suffix(target.suffix + ".embedded")
    if not embedded.is_file():
        raise FileNotFoundError(
            f"Artifact file and embedded payload are both missing: {artifact.relative_path}"
        )

    payload = embedded.read_bytes()
    if _sha256_bytes(payload) != artifact.sha256:
        raise ValueError(f"Embedded artifact checksum mismatch: {artifact.relative_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target


def materialize_repository_artifacts(root: str | Path) -> tuple[Path, ...]:
    """Materialize all manifest-declared embedded payloads under an artifact root."""

    artifact_root = Path(root)
    materialized: list[Path] = []
    for manifest_path in sorted(artifact_root.glob("*/manifest.json")):
        horizon_root = manifest_path.parent
        manifest = load_manifest(manifest_path)
        materialized.append(_materialize_file(horizon_root, manifest.model_file))
        materialized.append(_materialize_file(horizon_root, manifest.calibrator_file))
    return tuple(materialized)
