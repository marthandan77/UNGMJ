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
    fetch_eia_storage,
)
from ung_forecast.feed_status import (
    describe_context_feed,
    describe_price_feed,
    unavailable_feed,
)
from ung_forecast.providers import fetch_nasdaq_ung
from ung_forecast.rolling import generate_rolling_forecasts
from ung_forecast.scorecards import build_scorecards
from ung_forecast.weights import compute_model_weights

st.set_page_config(page_title="UNG Forecast", page_icon="📈", layout="wide")
st.title("UNG Rolling Forecast")
st.caption("Daily and five-session research forecasts. Not investment advice.")

run_report_path = Path("data/ledger/latest-run.json")
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
selected_models = st.multiselect(
    "Models to run",
    ["hist_gradient_boosting", "ridge"],
    default=["hist_gradient_boosting"],
    help="HistGradientBoosting is the production model. Select Ridge for a challenger run.",
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
        for row in forecasts if row["model_name"] == "ridge"
    }
    if chart_rows:
        st.subheader("Forecast range graphic — Ridge")
        st.line_chart(chart_rows, use_container_width=True)
        st.caption("The area between Forecast low and Forecast high is the expected price range; Forecast close is the middle estimate.")
    cards = build_scorecards(ForecastLedger().outcomes())
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
        st.subheader("Full accuracy history")
        outcomes = ForecastLedger().outcomes()
        history = pd.DataFrame(outcomes)
        if not history.empty:
            history["target_session"] = pd.to_datetime(history["target_session"])
            history = history.sort_values("target_session")
            history["model_horizon"] = history["model_name"] + " / " + history["horizon"].astype(str) + "d"
            history["cumulative_range_coverage"] = (
                history.groupby("model_horizon")["range_contains_day"].transform("mean")
            )
            history["rolling_20_range_coverage"] = (
                history.groupby("model_horizon")["range_contains_day"]
                .transform(lambda values: values.rolling(20, min_periods=1).mean())
            )
            st.line_chart(
                history.set_index("target_session").pivot(
                    columns="model_horizon", values="cumulative_range_coverage"
                ),
                y_label="Cumulative range coverage",
            )
            st.dataframe(
                history[["target_session", "model_name", "horizon", "close_error",
                         "range_contains_day", "cumulative_range_coverage",
                         "rolling_20_range_coverage"]],
                use_container_width=True, hide_index=True,
            )
    else:
        st.info("No matured forecasts yet; scorecards will populate as target sessions complete.")
st.caption("This is a research forecast, not financial advice.")
