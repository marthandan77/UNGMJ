from __future__ import annotations

import pytest

from ung_forecast.data.secrets import (
    normalize_provider_secrets,
    provider_secret_status,
    resolve_runtime_provider,
    secrets_fingerprint,
)


def test_normalizes_twelve_data_section_alias() -> None:
    secrets = normalize_provider_secrets({"twelve_data": {"api_key": "abc"}})
    assert secrets["twelvedata"]["api_key"] == "abc"


def test_normalizes_flat_twelve_data_api_key() -> None:
    secrets = normalize_provider_secrets({"TWELVEDATA_API_KEY": "abc"})
    assert secrets["twelvedata"]["api_key"] == "abc"


def test_twelve_data_status_requires_nonempty_key() -> None:
    configured, detail = provider_secret_status({"twelvedata": {"api_key": ""}}, "twelvedata")
    assert configured is False
    assert "empty" in detail


def test_twelve_data_status_accepts_nonempty_key() -> None:
    configured, detail = provider_secret_status(
        {"twelvedata": {"api_key": "abc"}},
        "twelvedata",
    )
    assert configured is True
    assert "detected" in detail


def test_runtime_provider_uses_explicit_streamlit_selection() -> None:
    provider = resolve_runtime_provider(
        "runtime",
        {
            "data": {"provider": "schwab"},
            "schwab": {
                "client_id": "id",
                "client_secret": "secret",
                "access_token": "token",
            },
        },
    )
    assert provider == "schwab"


def test_runtime_provider_infers_single_configured_provider() -> None:
    provider = resolve_runtime_provider(
        "runtime",
        {"twelvedata": {"api_key": "abc"}},
    )
    assert provider == "twelvedata"


def test_runtime_provider_defaults_to_yfinance_without_credentials() -> None:
    assert resolve_runtime_provider("runtime", {}) == "yfinance"


def test_runtime_provider_rejects_ambiguous_credentials() -> None:
    with pytest.raises(ValueError, match="Multiple provider credentials"):
        resolve_runtime_provider(
            "runtime",
            {
                "twelvedata": {"api_key": "abc"},
                "schwab": {
                    "client_id": "id",
                    "client_secret": "secret",
                    "access_token": "token",
                },
            },
        )


def test_secret_fingerprint_changes_when_value_changes() -> None:
    first = secrets_fingerprint({"data": {"provider": "twelvedata"}, "api": "one"})
    second = secrets_fingerprint({"data": {"provider": "twelvedata"}, "api": "two"})
    assert first != second
