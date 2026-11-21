"""Streamlit shell for the rolling UNG forecast system."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from ung_forecast.ledger import ForecastLedger
from ung_forecast.context_data import (
    combine_context_sources,
    fetch_cftc_positioning,
    fetch_eia_storage,
)
from ung_forecast.feed_status import (
    describe_context_feed,
    describe_price_feed,
    unavailable_feed,
)
from ung_forecast.providers import fetch_nasdaq_ung, fetch_yahoo_ung
from ung_forecast.rolling import generate_rolling_forecasts
from ung_forecast.scorecards import build_scorecards
from ung_forecast.weights import compute_model_weights
from ung_forecast.trade_desk import run_walk_forward_simulation

st.set_page_config(page_title="UNG Forecast", page_icon="📈", layout="wide")
st.title("UNG Rolling Forecast")
st.caption("Daily and five-session research forecasts. Not investment advice.")

run_report_path = Path("data/ledger/latest-run.json")

@st.cache_data(ttl=300, show_spinner=False)
def cached_scorecards() -> list[dict]:
    return build_scorecards(ForecastLedger().outcomes())

persisted_report = {}
if run_report_path.exists():
    try:
        persisted_report = json.loads(run_report_path.read_text(encoding="utf-8"))
        st.caption(
            f"Latest scan: {persisted_report.get('run_at', 'unknown')} | "
            f"Status: {persisted_report.get('status', 'unknown')} | "
            f"Source: {persisted_report.get('trigger', 'scheduled run')}"
        )
    except (OSError, json.JSONDecodeError):
        st.warning("The latest run report exists but could not be read.")

st.info(
    "Forecasts are provisional research outputs. After 20 matured outcomes per model and horizon, "
    "recent performance can inform the separately labeled adaptive consensus."
)

with st.expander("Adaptive Trade Desk simulation", expanded=False):
    st.write(
        "Simulates five factor-group experts over rolling historical forecast origins. "
        "Each horizon gets weights from previously matured out-of-sample errors."
    )
    st.caption(
        "Current inputs are UNG price, range, trend, momentum, and volume. Factor contributions "
        "show model associations, not causal effects."
    )
    if st.button("Run 1–5 session simulation", key="run_trade_desk_simulation"):
        with st.spinner("Running future-blind walk-forward simulation..."):
            try:
                try:
                    simulation_fetch = fetch_nasdaq_ung()
                except Exception:
                    simulation_fetch = fetch_yahoo_ung()
                simulation = run_walk_forward_simulation(
                    simulation_fetch.frame, min_train_rows=252, step=5, max_origins=126,
                )
                simulation["price_source"] = simulation_fetch.snapshot.provider
                simulation["data_through"] = simulation_fetch.snapshot.published_through.isoformat()
                sim_path = Path("data/ledger/trade-desk-simulation.json")
                sim_path.parent.mkdir(parents=True, exist_ok=True)
                sim_path.write_text(json.dumps(simulation, indent=2) + "\n", encoding="utf-8")
                st.session_state["trade_desk_simulation"] = simulation
            except Exception as simulation_exc:
                st.error(f"Simulation blocked: {type(simulation_exc).__name__}: {simulation_exc}")

    simulation = st.session_state.get("trade_desk_simulation")
    if simulation:
        st.caption(
            f"{simulation['simulation_origins']} historical origins | "
            f"{simulation['first_origin']} to {simulation['last_origin']} | "
            f"{simulation['bars']} daily bars"
        )
        if simulation["metrics_by_horizon"]:
            st.dataframe(pd.DataFrame(simulation["metrics_by_horizon"]), use_container_width=True, hide_index=True)
        if simulation["latest_forecasts"]:
            st.subheader("Current adaptive 1–5 session forecast")
            current = pd.DataFrame(simulation["latest_forecasts"])
            st.dataframe(
                current[["origin_session", "horizon", "reference_close", "predicted_low", "expected_close", "predicted_high"]].round(3),
                use_container_width=True,
                hide_index=True,
            )
            attribution = [
                {"Horizon": f"{row['horizon']} session(s)", "Factor": item["factor"],
                 "Adaptive weight": item["weight"],
                 "Close forecast contribution (%)": item["predicted_close_contribution_pct"]}
                for row in simulation["latest_forecasts"] for item in row["factor_contributions"]
            ]
            st.subheader("Factor weights and forecast contributions")
            st.dataframe(pd.DataFrame(attribution).round(4), use_container_width=True, hide_index=True)
        st.caption("The horizon range covers the path from the next session through the stated horizon.")

selected_models = st.multiselect(
    "Models to run",
    ["hist_gradient_boosting", "random_forest", "extra_trees"],
    default=["hist_gradient_boosting"],
    help="Select one or more tree-based models for comparison.",
)
if st.button("Scan", type="primary", use_container_width=True):
    with st.spinner("Scanning configured feeds, scoring completed forecasts, and updating models..."):
        try:
            fetched = fetch_nasdaq_ung()
            agreement = "Nasdaq primary"
            provider_status = "PRIMARY ONLY"
            data_feeds = [describe_price_feed("Nasdaq UNG", fetched, "Primary UNG OHLCV")]
            context_frames = []
            context_status = {}
            feed_specs = (
                (
                    "eia_storage", "EIA storage", fetch_eia_storage, ("storage_bcf",),
                    "https://www.eia.gov/dnav/ng/NG_STOR_WKLY_S1_W.htm",
                    "Historical workbook rows inherit the current workbook release date; do not treat it as historical point-in-time timing.",
                ),
                (
                    "cftc_positioning", "CFTC natural-gas positioning", fetch_cftc_positioning,
                    ("managed_money_net_pct_oi", "producer_merchant_net_pct_oi"),
                    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
                    "Weekly positions use a conservative release-time estimate; check holiday-week release timing before research use.",
                ),
            )
            for key, name, fetcher, value_columns, source_url, availability_note in feed_specs:
                try:
                    frame = fetcher()
                    context_frames.append(frame)
                    context_status[key] = "AVAILABLE"
                    data_feeds.append(describe_context_feed(
                        name, frame, value_columns=value_columns, availability_note=availability_note,
                        source_url=source_url,
                    ))
                except Exception as context_exc:
                    context_status[key] = f"UNAVAILABLE: {type(context_exc).__name__}"
                    data_feeds.append(unavailable_feed(
                        name, f"Fetch or validation failed ({type(context_exc).__name__})."
                    ))
            data_feeds.extend([
                unavailable_feed(
                    "Market depth", "No depth feed is configured in the daily UNG collector.",
                    status="NOT_CONNECTED",
                ),
                unavailable_feed(
                    "Weather / LNG / production", "No point-in-time provider is configured for these inputs.",
                    status="NOT_CONFIGURED",
                ),
            ])
            ledger = ForecastLedger()
            scored = ledger.score_matured(fetched.frame)
            generated = generate_rolling_forecasts(
                fetched.frame, context=combine_context_sources(context_frames),
                model_names=selected_models or ("hist_gradient_boosting",), ledger=ledger
            )
            report = {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "trigger": "manual_scan",
                "status": "READY" if context_frames else "PROVISIONAL",
                "provider": fetched.snapshot.provider,
                "provider_status": provider_status,
                "agreement": agreement,
                "context": context_status,
                "context_source_count": sum(value == "AVAILABLE" for value in context_status.values()),
                "data_feeds": data_feeds,
                "latest_session": str(fetched.frame.index[-1].date()),
                "history_rows": len(fetched.frame),
                "forecasts": len(generated),
                "scored": scored,
                "ledger_forecasts": len(ledger.forecasts()),
                "ledger_outcomes": len(ledger.outcomes()),
                "model_weights": compute_model_weights(ledger.outcomes()),
            }
            try:
                run_report_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_report_path = run_report_path.with_suffix(".json.tmp")
                temporary_report_path.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                temporary_report_path.replace(run_report_path)
            except OSError as report_exc:
                st.warning(
                    "Scan results are available in this app session; run report could not be saved "
                    f"({type(report_exc).__name__})."
                )
            st.session_state["latest_forecasts"] = generated
            st.session_state["latest_bars"] = fetched.frame.tail(120)
            st.session_state["run_quality"] = {
                "status": report["status"],
                "provider_status": provider_status,
                "agreement": agreement,
                "context_source_count": sum(value == "AVAILABLE" for value in context_status.values()),
                "data_feeds": data_feeds,
            }
            st.cache_data.clear()
            st.success(
                f"Scan complete: {len(data_feeds)} feed statuses refreshed, {scored} matured forecasts scored, "
                f"and {len(generated)} forecast rows updated."
            )
        except Exception as exc:
            st.error(f"Forecast blocked: {type(exc).__name__}: {exc}")

forecasts = st.session_state.get("latest_forecasts", [])
quality = st.session_state.get("run_quality")
feed_rows = quality.get("data_feeds") if quality else persisted_report.get("data_feeds", [])
st.subheader("Market input availability")
if feed_rows:
    feed_columns = {
        "source": "Source", "status": "Status", "latest_observation": "Latest observation",
        "latest_value": "Latest value", "available_at": "Availability timing",
        "retrieved_at": "Checked (UTC)", "source_url": "Source link", "role": "Use", "note": "Notes",
    }
    feed_table = pd.DataFrame(feed_rows)
    for column in feed_columns:
        if column not in feed_table:
            feed_table[column] = "—"
    st.dataframe(
        feed_table[list(feed_columns)].rename(columns=feed_columns),
        use_container_width=True,
        hide_index=True,
        column_config={"Source link": st.column_config.LinkColumn("Source link", display_text="Open")},
    )
    st.caption(
        "Feed availability means the source was fetched and validated. Henry Hub is reference-only; "
        "market depth and other unconfigured inputs are not forecast features."
    )
else:
    st.info("No persisted feed inventory yet. Run the daily collector or calculate a latest forecast to populate it.")
if quality:
    st.subheader("Status")
    q1, q2, q3 = st.columns(3)
    q1.metric("Status", quality["status"])
    q2.metric("Price data", "Available")
    q3.metric("Extra data", f"{quality.get('context_source_count', 0)}/2 sources")
if forecasts:
    st.subheader("Next five trading sessions")
    st.caption("Expected close is the main estimate. Low and high show the expected range.")
    forecast_columns = {
        "target_session": "Date", "predicted_low": "Low", "expected_close": "Expected close",
        "predicted_high": "High",
    }
    forecast_data = pd.DataFrame(forecasts)
    for column in forecast_columns:
        if column not in forecast_data:
            forecast_data[column] = "—"
    forecast_table = forecast_data[list(forecast_columns)].rename(columns=forecast_columns)
    for column in ("Low", "Expected close", "High"):
        forecast_table[column] = forecast_table[column].round(2)
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)
    chart_rows = {
        row["target_session"]: {
            "predicted_low": row["predicted_low"],
            "expected_close": row["expected_close"],
            "predicted_high": row["predicted_high"],
        }
        for row in forecasts if row["model_name"] == "hist_gradient_boosting"
    }
    if chart_rows:
        st.subheader("Forecast range graphic — HistGradientBoosting")
        st.line_chart(chart_rows, use_container_width=True)
        st.caption("The area between Forecast low and Forecast high is the expected price range; Forecast close is the middle estimate.")
    cards = cached_scorecards()
    st.subheader("Simple accuracy")
    if cards:
        scorecard_columns = [
            "model_name", "horizon", "matured_forecasts", "range_coverage", "status",
        ]
        st.dataframe(
            [{key: row.get(key) for key in scorecard_columns} for row in cards],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Coverage is the percentage of days where the actual high and low stayed inside the predicted range.")
        scorecard_frame = pd.DataFrame(cards)
        st.subheader("Range accuracy at a glance")
        chart_frame = scorecard_frame.pivot(index="model_name", columns="horizon", values="range_coverage")
        chart_frame.columns = [f"{int(value)}d" for value in chart_frame.columns]
        st.bar_chart(chart_frame, y_label="Range coverage (1.0 = 100%)")
    else:
        st.info("No matured forecasts yet; scorecards will populate as target sessions complete.")
st.caption("This is a research forecast, not financial advice.")
