"""Streamlit entry point.

The dashboard deliberately contains no quantitative formulas. It will render
validated forecast objects supplied by the core package in later stages.
"""

from __future__ import annotations

import streamlit as st

from ung_forecast.configuration import load_config
from ung_forecast.horizons import HORIZON_SPECS


config = load_config()

st.set_page_config(page_title=config.application_name, layout="wide")
st.title(config.application_name)
st.caption("Research-first, forecast-only prototype")

st.info(
    "The quantitative engines are under construction. No directional advice is available yet."
)

st.subheader("Locked forecast horizons")
for specification in HORIZON_SPECS.values():
    st.write(f"• {specification.display_name}: research status only")

with st.expander("Build identity"):
    st.code(f"Configuration hash: {config.configuration_hash}")
    st.code("Data provider: yfinance (prototype/research use)")
