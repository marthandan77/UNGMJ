"""Normalize runtime provider configuration without exposing secret values."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

SUPPORTED_PROVIDERS = frozenset({"twelvedata", "schwab", "yfinance"})
RUNTIME_PROVIDER_MARKERS = frozenset({"runtime", "auto", "secrets"})


def _plain_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        text_key = str(key)
        if isinstance(item, Mapping):
            result[text_key] = _plain_mapping(item)
        else:
            result[text_key] = item
    return result


def normalize_provider_secrets(raw_secrets: object) -> dict[str, Any]:
    """Return a canonical secrets mapping used by provider factories."""

    normalized = _plain_mapping(raw_secrets)

    if "twelvedata" not in normalized:
        for alias in ("twelve_data", "twelve-data", "twelveData"):
            section = normalized.get(alias)
            if isinstance(section, Mapping):
                normalized["twelvedata"] = dict(section)
                break

    if "twelvedata" not in normalized:
        flat_key = normalized.get("TWELVEDATA_API_KEY")
        if flat_key is None:
            flat_key = normalized.get("twelvedata_api_key")
        if flat_key is not None:
            section: dict[str, Any] = {"api_key": flat_key}
            base_url = normalized.get("TWELVEDATA_BASE_URL")
            if base_url is not None:
                section["base_url"] = base_url
            normalized["twelvedata"] = section

    return normalized


def secrets_fingerprint(secrets: Mapping[str, Any]) -> str:
    """Return a one-way cache identity without logging secret values."""

    serialized = json.dumps(
        _plain_mapping(secrets),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _explicit_provider(secrets: Mapping[str, Any]) -> str | None:
    data_section = secrets.get("data")
    if isinstance(data_section, Mapping):
        value = str(data_section.get("provider", "")).strip().lower()
        if value:
            return value
    for key in ("DATA_PROVIDER", "data_provider"):
        value = str(secrets.get(key, "")).strip().lower()
        if value:
            return value
    return None


def resolve_runtime_provider(
    configured_provider: str,
    secrets: Mapping[str, Any],
) -> str:
    """Resolve provider without requiring a GitHub configuration edit.

    A concrete provider in ``config.yaml`` remains supported for local research.
    The ``runtime`` marker delegates selection to Streamlit Secrets.
    """

    configured = configured_provider.strip().lower()
    if configured not in RUNTIME_PROVIDER_MARKERS:
        if configured not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported configured provider: {configured_provider}")
        return configured

    explicit = _explicit_provider(secrets)
    if explicit is not None:
        if explicit not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported runtime provider: {explicit}")
        return explicit

    ready: list[str] = []
    for candidate in ("twelvedata", "schwab"):
        is_ready, _ = provider_secret_status(secrets, candidate)
        if is_ready:
            ready.append(candidate)
    if len(ready) == 1:
        return ready[0]
    if len(ready) > 1:
        raise ValueError(
            "Multiple provider credentials detected; set [data] provider explicitly"
        )
    return "yfinance"


def provider_secret_status(
    secrets: Mapping[str, Any],
    provider_name: str,
) -> tuple[bool, str]:
    """Return configuration readiness and a non-secret diagnostic message."""

    normalized_name = provider_name.lower()
    if normalized_name == "yfinance":
        return True, "No secret required"
    section = secrets.get(normalized_name)
    if not isinstance(section, Mapping):
        return False, f"Missing [{normalized_name}] section"

    if normalized_name == "twelvedata":
        api_key = str(section.get("api_key", "")).strip()
        if not api_key:
            return False, "[twelvedata] exists but api_key is empty"
        return True, "Twelve Data API key detected"

    if normalized_name == "schwab":
        client_id = str(section.get("client_id", "")).strip()
        client_secret = str(section.get("client_secret", "")).strip()
        token_present = bool(
            str(section.get("access_token", "")).strip()
            or str(section.get("refresh_token", "")).strip()
        )
        if not client_id or not client_secret or not token_present:
            return False, "[schwab] is missing client credentials or token"
        return True, "Schwab credentials detected"

    return False, f"Unsupported provider: {provider_name}"
