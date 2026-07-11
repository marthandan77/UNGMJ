"""Streamlit entry point.

The dashboard contains no quantitative formulas. It configures the runtime
provider and renders validated objects from the core package.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import streamlit as st
from ung_forecast.configuration import load_config
from ung_forecast.data import MarketDataProvider, build_market_data_provider
from ung_forecast.horizons import HORIZON_SPECS


config = load_config()


def load_runtime_secrets() -> dict[str, Any]:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


runtime_secrets = load_runtime_secrets()


@st.cache_resource
def runtime_provider(configuration_hash: str) -> MarketDataProvider:
    del configuration_hash
    return build_market_data_provider(config, secrets=runtime_secrets)


st.set_page_config(page_title=config.application_name, layout="wide")
st.title(config.application_name)
st.caption("GitHub + Streamlit forecast research build")

with st.sidebar:
    st.header("System")
    st.write(f"Provider: **{config.data.provider.upper()}**")
    st.write(f"Primary symbol: **{config.data.primary_symbol}**")
    st.write("Mode: **Research / Shadow only**")
    st.caption("The application does not place or route orders.")

st.subheader("Data connection")
if "schwab" not in runtime_secrets:
    st.warning(
        "Schwab is not configured. Add the [schwab] values from "
        ".streamlit/secrets.toml.example to Streamlit App settings > Secrets."
    )
else:
    st.success("Schwab secrets are present. Values are not displayed or logged.")
    if st.button("Test Schwab UNG feed", type="primary"):
        try:
            provider = runtime_provider(config.configuration_hash)
            bundle = provider.download(
                symbol=config.data.primary_symbol,
                interval="5m",
                period="",
                as_of=datetime.now(UTC),
            )
        except Exception as exc:  # Streamlit must surface provider failures safely.
            st.error(f"Schwab feed test failed: {type(exc).__name__}: {exc}")
        else:
            latest = bundle.frame.iloc[-1]
            st.success("Schwab market data passed validation.")
            first, second, third = st.columns(3)
            first.metric("Latest close", f"${float(latest['Close']):.2f}")
            second.metric("Validated bars", f"{len(bundle.frame):,}")
            third.metric("Latest bar", str(bundle.frame.index[-1]))
            with st.expander("Validated data preview"):
                st.dataframe(bundle.frame.tail(20), use_container_width=True)

st.subheader("Forecast horizons")
for specification in HORIZON_SPECS.values():
    st.write(f"• {specification.display_name}: research status only")

st.info(
    "Directional forecasts remain disabled until a horizon passes its purged "
    "walk-forward statistical approval tests."
)

with st.expander("Build identity"):
    st.code(f"Configuration hash: {config.configuration_hash}")
    st.code(f"Configured provider: {config.data.provider}")
    st.code("Persistence: local prototype; PostgreSQL deferred")
