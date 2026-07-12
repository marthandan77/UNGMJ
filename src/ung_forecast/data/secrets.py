"""Normalize provider credentials from Streamlit Secrets without exposing values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
    """Return a canonical secrets mapping used by provider factories.

    Accepted Twelve Data forms:
    - [twelvedata] api_key = "..."
    - [twelve_data] api_key = "..."
    - TWELVEDATA_API_KEY = "..."
    """

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
