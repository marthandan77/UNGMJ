"""Streamlit shell for the rolling UNG forecast system."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ung_forecast.ledger import ForecastLedger
from ung_forecast.context_data import (
    combine_context_sources,
    fetch_cftc_positioning,
    fetch_eia_henry_hub_spot,
    fetch_eia_storage,
)
from ung_forecast.feed_status import (
    describe_context_feed,
    describe_henry_hub_feed,
    describe_price_feed,
    unavailable_feed,
)
from ung_forecast.providers import compare_sources, fetch_nasdaq_ung, fetch_yahoo_ung
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
            f"Last scheduled run: {persisted_report.get('run_at', 'unknown')} | "
            f"Status: {persisted_report.get('status', 'unknown')}"
        )
    except (OSError, json.JSONDecodeError):
        st.warning("The latest run report exists but could not be read.")

st.info(
    "Forecasts are provisional research outputs. After 20 matured outcomes per model and horizon, "
    "recent performance can inform the separately labeled adaptive consensus."
)
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
            # Nasdaq is the primary source. Yahoo is an independent health check,
            # but rate limiting there must not prevent a valid primary forecast.
            try:
                confirmation = fetch_yahoo_ung()
                agreement = compare_sources(fetched, confirmation)
                provider_status = "CONFIRMED"
            except Exception as yahoo_exc:
                agreement = f"Nasdaq primary accepted; Yahoo confirmation unavailable: {yahoo_exc}"
                provider_status = "PRIMARY ONLY"
                st.warning(agreement)
                confirmation = None
            data_feeds = [describe_price_feed("Nasdaq UNG", fetched, "Primary UNG OHLCV")]
            if confirmation is not None:
                data_feeds.append(describe_price_feed(
                    "Yahoo UNG", confirmation, "Independent price confirmation"
                ))
            else:
                data_feeds.append(unavailable_feed(
                    "Yahoo UNG", "Price confirmation unavailable; primary source may still be used."
                ))
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
                    "https://www.cftc.gov/dea/futures/nat_gas_lf.htm",
                    "Availability uses a conservative report-date + 4 days, 21:00 UTC rule, not an observed publication timestamp.",
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
            try:
                data_feeds.append(describe_henry_hub_feed(fetch_eia_henry_hub_spot()))
            except Exception as henry_hub_exc:
                data_feeds.append(unavailable_feed(
                    "EIA Henry Hub spot", f"Fetch or validation failed ({type(henry_hub_exc).__name__})."
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
            ledger.score_matured(fetched.frame)
            generated = generate_rolling_forecasts(
                fetched.frame, context=combine_context_sources(context_frames),
                model_names=selected_models or ("ridge",), ledger=ledger
            )
            st.session_state["latest_forecasts"] = generated
            st.session_state["latest_bars"] = fetched.frame.tail(120)
            st.session_state["run_quality"] = {
                "status": "READY" if len(context_frames) == 2 and provider_status == "CONFIRMED" else "PROVISIONAL",
                "provider_status": provider_status,
                "agreement": agreement,
                "context_source_count": sum(value == "AVAILABLE" for value in context_status.values()),
                "data_feeds": data_feeds,
            }
            st.success(f"Generated {len(generated)} individual model forecasts.")
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
    st.subheader("Latest run quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Status", quality["status"])
    q2.metric("UNG providers", quality.get("provider_status", "UNKNOWN"))
    q3.metric("Context sources", f"{quality.get('context_source_count', 0)}/2")
    st.caption(quality["agreement"])
if forecasts:
    st.subheader("Next five trading sessions")
    st.caption("Low and high show the model's expected trading range. Close is the expected closing price.")
    forecast_columns = {
        "target_session": "Session", "model_name": "Model", "horizon": "Days ahead",
        "predicted_low": "Forecast low", "expected_close": "Forecast close",
        "predicted_high": "Forecast high", "probability_up": "Up probability",
        "calibration_status": "Range calibration",
        "calibration_sample_count": "Calibration samples",
        "learning_sample_count": "Consensus samples",
    }
    forecast_data = pd.DataFrame(forecasts)
    for column in forecast_columns:
        if column not in forecast_data:
            forecast_data[column] = "—"
    forecast_table = forecast_data[list(forecast_columns)].rename(columns=forecast_columns)
    forecast_table["Up probability"] = (forecast_table["Up probability"] * 100).round(0).astype(str) + "%"
    for column in ("Forecast low", "Forecast close", "Forecast high"):
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
    st.subheader("Individual model scorecards")
    if cards:
        scorecard_columns = [
            "model_name", "horizon", "matured_forecasts", "range_coverage",
            "mean_abs_low_error", "mean_abs_high_error", "average_range_width", "status",
        ]
        st.dataframe(
            [{key: row.get(key) for key in scorecard_columns} for row in cards],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Range coverage = percentage of sessions where the actual low and high stayed inside the forecast range.")
        st.caption("Lower low/high errors are better. Narrower ranges are better only when coverage remains high.")
        scorecard_frame = pd.DataFrame(cards)
        st.subheader("Range accuracy at a glance")
        chart_frame = scorecard_frame.pivot(index="model_name", columns="horizon", values="range_coverage")
        chart_frame.columns = [f"{int(value)}d" for value in chart_frame.columns]
        st.bar_chart(chart_frame, y_label="Range coverage (1.0 = 100%)")
    else:
        st.info("No matured forecasts yet; scorecards will populate as target sessions complete.")
    weights = compute_model_weights(ForecastLedger().outcomes())
    if weights:
        st.subheader("Learned model weights")
        st.dataframe(weights, use_container_width=True, hide_index=True)
        st.caption(
            "Weights use inverse close error from the most recent matured forecasts, separately by horizon. "
            "At least 20 outcomes are required per model and horizon; the adaptive consensus is scored separately."
        )
    else:
        st.info("Model weights remain neutral until sufficient matured out-of-sample evidence exists.")

st.subheader("Planned output")
st.write(
    "Each run keeps individual model forecasts visible. Once enough matured evidence exists, "
    "a separately labeled adaptive consensus uses recent model scores; its outcomes do not train its own weights."
)
