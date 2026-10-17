"""Streamlit shell for the rolling UNG forecast system."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ung_forecast.ledger import ForecastLedger
from ung_forecast.context_data import combine_context_sources, fetch_cftc_positioning, fetch_eia_storage
from ung_forecast.providers import compare_sources, fetch_nasdaq_ung, fetch_yahoo_ung
from ung_forecast.rolling import generate_rolling_forecasts
from ung_forecast.scorecards import build_scorecards
from ung_forecast.weights import compute_model_weights

st.set_page_config(page_title="UNG Forecast", page_icon="📈", layout="wide")
st.title("UNG Rolling Forecast")
st.caption("Daily and five-session research forecasts. Not investment advice.")

run_report_path = Path("data/ledger/latest-run.json")
if run_report_path.exists():
    try:
        persisted_report = json.loads(run_report_path.read_text(encoding="utf-8"))
        st.caption(
            f"Last scheduled run: {persisted_report.get('run_at', 'unknown')} | "
            f"Status: {persisted_report.get('status', 'unknown')}"
        )
    except (OSError, json.JSONDecodeError):
        st.warning("The latest run report exists but could not be read.")

st.info("Forecasts are provisional research outputs until enough matured scorecard history exists.")
selected_models = st.multiselect(
    "Models to run",
    ["ridge", "hist_gradient_boosting", "random_forest", "extra_trees"],
    default=["ridge"],
    help="Ridge is fastest. Tree challengers provide independent scorecards but take longer.",
)
if st.button("Calculate latest forecast", type="primary", use_container_width=True):
    with st.spinner("Fetching validated UNG data and calculating individual forecasts..."):
        try:
            fetched = fetch_nasdaq_ung()
            confirmation = fetch_yahoo_ung()
            agreement = compare_sources(fetched, confirmation)
            context_frames = []
            context_status = {}
            for name, fetcher in (("EIA storage", fetch_eia_storage), ("CFTC positioning", fetch_cftc_positioning)):
                try:
                    context_frames.append(fetcher())
                    context_status[name] = "AVAILABLE"
                except Exception as context_exc:
                    context_status[name] = f"UNAVAILABLE: {context_exc}"
            ledger = ForecastLedger()
            generated = generate_rolling_forecasts(
                fetched.frame, context=combine_context_sources(context_frames),
                model_names=selected_models or ("ridge",), ledger=ledger
            )
            ledger.score_matured(fetched.frame)
            st.session_state["latest_forecasts"] = generated
            st.session_state["latest_bars"] = fetched.frame.tail(120)
            st.session_state["run_quality"] = {
                "status": "READY" if len(context_frames) == 2 else "PROVISIONAL",
                "agreement": agreement,
                **context_status,
            }
            st.success(f"Generated {len(generated)} individual model forecasts.")
        except Exception as exc:
            st.error(f"Forecast blocked: {type(exc).__name__}: {exc}")

forecasts = st.session_state.get("latest_forecasts", [])
quality = st.session_state.get("run_quality")
if quality:
    st.subheader("Latest run quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Status", quality["status"])
    q2.metric("UNG providers", "CONFIRMED")
    q3.metric("Context sources", f"{sum(value == 'AVAILABLE' for key, value in quality.items() if key not in ('status', 'agreement'))}/2")
    st.caption(quality["agreement"])
    for key, value in quality.items():
        if key not in ("status", "agreement"):
            st.write(f"**{key}:** {value}")
if forecasts:
    st.subheader("Next five trading sessions")
    st.dataframe(forecasts, use_container_width=True, hide_index=True)
    chart_rows = {
        row["target_session"]: {
            "predicted_low": row["predicted_low"],
            "expected_close": row["expected_close"],
            "predicted_high": row["predicted_high"],
        }
        for row in forecasts if row["model_name"] == "ridge"
    }
    if chart_rows:
        st.subheader("Ridge rolling range forecast")
        st.line_chart(chart_rows, use_container_width=True)
    cards = build_scorecards(ForecastLedger().outcomes())
    st.subheader("Individual model scorecards")
    if cards:
        st.dataframe(cards, use_container_width=True, hide_index=True)
    else:
        st.info("No matured forecasts yet; scorecards will populate as target sessions complete.")
    weights = compute_model_weights(ledger.outcomes())
    if weights:
        st.subheader("Learned model weights")
        st.dataframe(weights, use_container_width=True, hide_index=True)
    else:
        st.info("Model weights remain neutral until sufficient matured out-of-sample evidence exists.")

st.subheader("Planned output")
st.write("Each run displays separate forecasts for each model and horizon; no model is hidden behind an ensemble.")
