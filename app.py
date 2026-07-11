"""Streamlit entry point.

The dashboard contains no quantitative formulas. It configures the runtime
provider and renders validated objects from the core package.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import streamlit as st

from ung_forecast.configuration import load_config
from ung_forecast.data import (
    MarketDataProvider,
    ParquetCache,
    build_market_data_provider,
    load_runtime_market_data,
)
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
    if st.button("Load and validate Schwab data", type="primary"):
        try:
            provider = runtime_provider(config.configuration_hash)
            runtime_data = load_runtime_market_data(
                provider,
                ParquetCache(config.data.cache_directory),
                symbol=config.data.primary_symbol,
                as_of=datetime.now(UTC),
                allow_cache_fallback=True,
            )
        except Exception as exc:  # Streamlit must surface provider failures safely.
            st.error(f"Schwab data load failed: {type(exc).__name__}: {exc}")
        else:
            if not runtime_data.bundles_by_interval:
                st.error("No validated live or cached market data is available.")
            else:
                st.success("Runtime market data load completed.")
                rows: list[dict[str, object]] = []
                for interval, bundle in runtime_data.bundles_by_interval.items():
                    latest = bundle.frame.iloc[-1]
                    rows.append(
                        {
                            "Interval": interval,
                            "Source": runtime_data.source_by_interval[interval],
                            "Bars": len(bundle.frame),
                            "Latest close": round(float(latest["Close"]), 4),
                            "Latest timestamp": str(bundle.frame.index[-1]),
                            "Provider": bundle.provenance.provider,
                        }
                    )
                st.dataframe(rows, use_container_width=True, hide_index=True)
                if runtime_data.errors_by_interval:
                    with st.expander("Provider errors and cache fallbacks"):
                        for interval, error in runtime_data.errors_by_interval.items():
                            st.warning(f"{interval}: {error}")
                with st.expander("Latest validated 5-minute bars"):
                    five_minute = runtime_data.bundles_by_interval.get("5m")
                    if five_minute is not None:
                        st.dataframe(five_minute.frame.tail(20), use_container_width=True)
                    else:
                        st.write("Five-minute data is unavailable.")

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
