"""Streamlit entry point.

The dashboard contains no quantitative formulas. It configures the runtime
provider and renders validated objects from the core package.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ung_forecast.artifacts import ArtifactState, discover_horizon_artifacts
from ung_forecast.configuration import load_config
from ung_forecast.dashboard import build_artifact_status_rows
from ung_forecast.data import (
    MarketDataProvider,
    ParquetCache,
    RuntimeDataSet,
    build_market_data_provider,
    load_runtime_market_data,
)
from ung_forecast.data.secrets import normalize_provider_secrets, provider_secret_status
from ung_forecast.horizons import HORIZON_SPECS
from ung_forecast.runtime_forecast import generate_runtime_probability_forecast

config = load_config()
ARTIFACT_ROOT = Path("artifacts/models")


def load_runtime_secrets() -> dict[str, Any]:
    try:
        return normalize_provider_secrets(st.secrets)
    except Exception:
        return {}


runtime_secrets = load_runtime_secrets()


@st.cache_resource
def runtime_provider(configuration_hash: str, secrets_fingerprint: tuple[str, ...]) -> MarketDataProvider:
    del configuration_hash, secrets_fingerprint
    return build_market_data_provider(config, secrets=runtime_secrets)


st.set_page_config(page_title=config.application_name, layout="wide")
st.title(config.application_name)
st.caption("GitHub + Streamlit forecast research build")

provider_name = config.data.provider.lower()
provider_label = {
    "twelvedata": "Twelve Data",
    "schwab": "Schwab",
    "yfinance": "yfinance",
}.get(provider_name, config.data.provider)
provider_configured, provider_status = provider_secret_status(runtime_secrets, provider_name)
secret_fingerprint = tuple(sorted(str(key) for key in runtime_secrets))

with st.sidebar:
    st.header("System")
    st.write(f"Provider: **{provider_label}**")
    st.write(f"Primary symbol: **{config.data.primary_symbol}**")
    st.write("Mode: **Research / Shadow only**")
    st.caption("The application does not place or route orders.")

st.subheader("Data connection")
if not provider_configured:
    st.warning(f"{provider_label} is not configured: {provider_status}.")
    with st.expander("Secret diagnostics"):
        st.write(f"Detected top-level secret names: {', '.join(secret_fingerprint) or 'none'}")
        st.caption("Secret values are never displayed.")
        st.code(
            '[twelvedata]\napi_key = "YOUR_REAL_API_KEY"\nbase_url = "https://api.twelvedata.com"'
        )
else:
    st.success(f"{provider_status}. Values are not displayed or logged.")
    if st.button(f"Load and validate {provider_label} data", type="primary"):
        try:
            provider = runtime_provider(config.configuration_hash, secret_fingerprint)
            runtime_data = load_runtime_market_data(
                provider,
                ParquetCache(config.data.cache_directory),
                symbol=config.data.primary_symbol,
                as_of=datetime.now(UTC),
                allow_cache_fallback=True,
            )
        except Exception as exc:
            st.error(f"{provider_label} data load failed: {type(exc).__name__}: {exc}")
        else:
            st.session_state["runtime_data"] = runtime_data

runtime_data_state = st.session_state.get("runtime_data")
if isinstance(runtime_data_state, RuntimeDataSet):
    if not runtime_data_state.bundles_by_interval:
        st.error("No validated live or fresh cached market data is available.")
    else:
        st.success("Runtime market data load completed.")
        data_rows: list[dict[str, object]] = []
        for interval, bundle in runtime_data_state.bundles_by_interval.items():
            latest = bundle.frame.iloc[-1]
            data_rows.append(
                {
                    "Interval": interval,
                    "Source": runtime_data_state.source_by_interval[interval],
                    "Bars": len(bundle.frame),
                    "Latest close": round(float(latest["Close"]), 4),
                    "Latest timestamp": str(bundle.frame.index[-1]),
                    "Provider": bundle.provenance.provider,
                }
            )
        st.dataframe(data_rows, use_container_width=True, hide_index=True)
        if runtime_data_state.errors_by_interval:
            with st.expander("Provider errors and cache fallbacks"):
                for interval, error in runtime_data_state.errors_by_interval.items():
                    st.warning(f"{interval}: {error}")

st.subheader("Model artifact readiness")
artifact_results = discover_horizon_artifacts(
    ARTIFACT_ROOT,
    expected_configuration_hash=config.quantitative_configuration_hash,
    comparison_available=False,
)
artifact_rows = build_artifact_status_rows(artifact_results)
st.dataframe(
    [
        {
            "Horizon": row.horizon,
            "State": row.state,
            "Model": row.model_version,
            "Features": row.feature_version,
            "Samples": row.samples,
            "Brier": row.brier_score,
            "Calibration error": row.calibration_error,
            "Detail": row.detail,
        }
        for row in artifact_rows
    ],
    use_container_width=True,
    hide_index=True,
)

validated_count = sum(row.state == "VALIDATED" for row in artifact_rows)
if validated_count == 0:
    st.info(
        "No checksum-verified, statistically approved and runtime-compatible horizon "
        "artifact is installed. Directional forecasts remain disabled."
    )
else:
    st.success(f"{validated_count} horizon artifact(s) passed integrity and compatibility checks.")

if isinstance(runtime_data_state, RuntimeDataSet) and validated_count:
    st.subheader("Research probability forecasts")
    probability_rows: list[dict[str, object]] = []
    for horizon in HORIZON_SPECS:
        artifact_result = artifact_results[horizon]
        if artifact_result.state is not ArtifactState.VALIDATED:
            continue
        try:
            forecast = generate_runtime_probability_forecast(
                artifact_result,
                runtime_data_state,
            )
        except Exception as exc:
            probability_rows.append(
                {
                    "Horizon": HORIZON_SPECS[horizon].display_name,
                    "Status": "UNAVAILABLE",
                    "Detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        probability_rows.append(
            {
                "Horizon": HORIZON_SPECS[horizon].display_name,
                "Status": "RESEARCH STATUS ONLY",
                "Lower first": round(100.0 * forecast.probabilities.lower_first, 2),
                "Upper first": round(100.0 * forecast.probabilities.upper_first, 2),
                "Neither": round(100.0 * forecast.probabilities.neither, 2),
                "Current price": round(forecast.current_price, 4),
                "As of": str(forecast.timestamp),
                "Source": forecast.data_source,
                "Model": forecast.model_version,
            }
        )
    if probability_rows:
        st.dataframe(probability_rows, use_container_width=True, hide_index=True)
        st.warning(
            "These are statistical research probabilities, not buy or sell advice. "
            "Trading approval and shadow evidence are not yet complete."
        )

with st.expander("Build identity"):
    st.code(f"Operational configuration hash: {config.configuration_hash}")
    st.code(f"Quantitative configuration hash: {config.quantitative_configuration_hash}")
    st.code(f"Configured provider: {config.data.provider}")
    st.code(f"Artifact root: {ARTIFACT_ROOT}")
    st.code("Local cache and ledger are temporary on Streamlit Community Cloud.")
