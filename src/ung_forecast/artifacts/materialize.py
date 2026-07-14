"""Materialize repository-embedded model payloads with checksum verification."""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from .manifest import ArtifactFile


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_artifact_file(horizon_root: Path, artifact: ArtifactFile) -> Path:
    """Materialize one base64 payload when its binary file is absent or invalid."""

    target = horizon_root / artifact.relative_path
    target_exists = target.is_file()
    if target_exists and _sha256_file(target) == artifact.sha256:
        return target

    embedded = target.with_suffix(target.suffix + ".b64")
    if not embedded.is_file():
        if target_exists:
            raise ValueError(f"Checksum mismatch: {artifact.relative_path}")
        raise FileNotFoundError(
            f"Artifact file and embedded payload are both missing: {artifact.relative_path}"
        )

    try:
        payload = base64.b64decode(embedded.read_text(encoding="ascii"), validate=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Embedded artifact payload is invalid: {artifact.relative_path}") from exc
    if _sha256_bytes(payload) != artifact.sha256:
        raise ValueError(f"Embedded artifact checksum mismatch: {artifact.relative_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target
