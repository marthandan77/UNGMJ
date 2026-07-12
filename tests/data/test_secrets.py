from __future__ import annotations

from ung_forecast.data.secrets import normalize_provider_secrets, provider_secret_status


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
