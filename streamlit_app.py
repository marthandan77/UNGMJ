"""Interactive quantitative trade desk for UNG."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ung_forecast.context_data import combine_context_sources, fetch_cftc_positioning, fetch_eia_storage
from ung_forecast.feed_status import describe_context_feed, describe_price_feed, unavailable_feed
from ung_forecast.features import FORECAST_FEATURE_GROUPS
from ung_forecast.ledger import ForecastLedger
from ung_forecast.outside_rth import build_current_outside_rth_snapshot, outside_rth_frame
from ung_forecast.providers import fetch_alpaca_ung_daily, fetch_alpaca_ung_intraday
from ung_forecast.rolling import generate_rolling_forecasts
from ung_forecast.scorecards import build_scorecards
from ung_forecast.trade_desk import run_walk_forward_simulation
from ung_forecast.weights import compute_model_weights


st.set_page_config(page_title="UNG Quant Desk", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.4rem; padding-bottom:3rem; max-width:1500px;}
h1,h2,h3 {letter-spacing:-.025em;}
[data-testid="stMetric"] {background:#111a27; border:1px solid #243247; padding:14px 16px; border-radius:12px;}
[data-testid="stMetricLabel"] {color:#9fb0c6;}
[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;}
.desk-subtitle {color:#91a2b9; margin-top:-.6rem;}
.desk-pill {display:inline-block; border:1px solid #2c4058; background:#111a27; color:#b9c8db; padding:4px 10px; border-radius:999px; font-size:.78rem; margin-right:6px;}
</style>
""", unsafe_allow_html=True)

LEDGER_DIR = Path("data/ledger")
REPORT_PATH = LEDGER_DIR / "latest-run.json"
DATA_BUDGET_BYTES = 8 * 1024 * 1024


def latest_vintage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    issue = max(str(row.get("issued_at", "")) for row in rows)
    return sorted((row for row in rows if str(row.get("issued_at", "")) == issue), key=lambda row: int(row.get("horizon", 0)))


def read_report() -> dict[str, Any]:
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def storage_bytes() -> int:
    return sum(path.stat().st_size for path in LEDGER_DIR.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl"}) if LEDGER_DIR.exists() else 0


def money(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"${float(value):,.2f}"


def consensus_for(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("model_name") == "dma_consensus" and int(row.get("horizon", 0)) == horizon), None)


def confidence_frame(scorecards: list[dict[str, Any]], horizon: int) -> pd.DataFrame:
    rows = [{
        "Model": row["model_name"].replace("hgb_expert_", "").replace("hist_gradient_boosting", "HGB · all features").replace("dma_consensus", "DMA · blended HGB"),
        "Historical confidence (%)": 100 * float(row["range_coverage"]),
        "Completed forecasts": int(row["matured_forecasts"]),
        "Evidence status": row["status"],
    } for row in scorecards if int(row["horizon"]) == horizon]
    return pd.DataFrame(rows).sort_values("Historical confidence (%)", ascending=False) if rows else pd.DataFrame()


def prior_consensus(ledger: ForecastLedger, forecast: dict[str, Any]) -> dict[str, Any] | None:
    earlier = [row for row in ledger.forecasts()
        if row.get("model_name") == "dma_consensus"
        and row.get("target_session") == forecast.get("target_session")
        and int(row.get("horizon", 0)) == int(forecast.get("horizon", 0))
        and str(row.get("issued_at", "")) < str(forecast.get("issued_at", ""))]
    return max(earlier, key=lambda row: str(row.get("issued_at", ""))) if earlier else None


def run_scan(ledger: ForecastLedger) -> None:
    with st.spinner("Fetching prices and updating the adaptive forecast…"):
        fetched = fetch_alpaca_ung_daily()
        feeds = [describe_price_feed("Alpaca UNG daily SIP", fetched, "Daily range model input")]
        contexts, context_status = [], {}
        specs = (
            ("eia_storage", "EIA storage", fetch_eia_storage, ("storage_bcf",), "https://www.eia.gov/dnav/ng/NG_STOR_WKLY_S1_W.htm", "Older workbook rows do not preserve every release vintage."),
            ("cftc_positioning", "CFTC natural-gas positioning", fetch_cftc_positioning, ("managed_money_net_pct_oi", "producer_merchant_net_pct_oi"), "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm", "Release timing is estimated for holiday weeks."),
        )
        for key, name, fetcher, columns, url, note in specs:
            try:
                frame = fetcher()
                contexts.append(frame)
                context_status[key] = "AVAILABLE"
                feeds.append(describe_context_feed(name, frame, value_columns=columns, availability_note=note, source_url=url))
            except Exception as exc:
                context_status[key] = f"UNAVAILABLE: {type(exc).__name__}"
                feeds.append(unavailable_feed(name, f"Fetch failed ({type(exc).__name__})."))
        feeds.extend([
            unavailable_feed("NG futures curve", "No contract-level feed is configured.", status="NOT_CONFIGURED"),
            unavailable_feed("Weather / LNG / production", "No point-in-time provider is configured.", status="NOT_CONFIGURED"),
            unavailable_feed("Quotes / order book", "Bars do not provide full market depth.", status="NOT_CONNECTED"),
        ])
        scored = ledger.score_matured(fetched.frame)
        outside_status, outside_slot = "UNAVAILABLE", None
        try:
            intraday = fetch_alpaca_ung_intraday()
            snapshot = build_current_outside_rth_snapshot(intraday.frame, fetched.frame)
            snapshot["retrieved_at"] = intraday.snapshot.retrieved_at.isoformat()
            ledger.append_outside_rth_feature(snapshot)
            outside_slot, outside_status = snapshot["snapshot_slot"], "AVAILABLE"
            prefix = "mid" if outside_slot == "mid" else "1h"
            feeds.append({"source":"Alpaca UNG 1-minute SIP extended session", "status":outside_status,
                "latest_observation":snapshot["target_session"],
                "latest_value":f"{money(snapshot[f'ung_premarket_{prefix}_last_price'])} · return {float(snapshot[f'ung_premarket_{prefix}_return']):+.3%}",
                "available_at":snapshot["available_at"], "retrieved_at":snapshot["retrieved_at"],
                "rows":len(intraday.frame), "role":f"{outside_slot} pre-open and prior after-hours features",
                "source_url":intraday.snapshot.source_url, "note":"Consolidated trade bars, delayed 15 minutes; no quote depth."})
        except Exception as exc:
            outside_status = f"UNAVAILABLE: {type(exc).__name__}"
            feeds.append(unavailable_feed("Alpaca UNG 1-minute SIP extended session", f"Snapshot unavailable ({type(exc).__name__}); scheduled fetches run at 7:02 and 8:47 a.m. ET."))
        generated = generate_rolling_forecasts(fetched.frame, context=combine_context_sources(contexts),
            outside_rth=outside_rth_frame(ledger.outside_rth_features()), ledger=ledger)
        storage = ledger.purge()
        report = {"run_at":datetime.now(timezone.utc).isoformat(), "trigger":"manual_scan",
            "status":"READY" if contexts else "PROVISIONAL", "provider":"alpaca",
            "agreement":"Alpaca consolidated SIP", "context":context_status,
            "outside_rth":outside_status, "outside_rth_slot":outside_slot, "data_feeds":feeds,
            "latest_session":fetched.frame.index[-1].date().isoformat(),
            "reference_close":float(fetched.frame.iloc[-1].close), "history_rows":len(fetched.frame),
            "forecasts":len(generated), "scored":scored,
            "ledger_forecasts":len(ledger.forecasts()), "ledger_outcomes":len(ledger.outcomes()),
            "storage":storage, "model_weights":compute_model_weights(ledger.outcomes())}
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        st.session_state["latest_forecasts"] = latest_vintage(generated)
        st.session_state["latest_bars"] = fetched.frame.tail(180)
        st.session_state["run_report"] = report
        st.cache_data.clear()


ledger = ForecastLedger()
report = st.session_state.get("run_report") or read_report()
if "latest_forecasts" not in st.session_state:
    st.session_state["latest_forecasts"] = latest_vintage(ledger.forecasts())
forecasts = st.session_state["latest_forecasts"]
outcomes = ledger.outcomes()
scorecards = [row for row in build_scorecards(outcomes)
    if row["model_name"] in {"hist_gradient_boosting", "dma_consensus"}
    or str(row["model_name"]).startswith("hgb_expert_")]

with st.sidebar:
    st.markdown("### ◈ UNG Quant Desk")
    st.caption("Range forecasting · Adaptive learning · Trade planning")
    st.divider()
    horizon = st.selectbox("Forecast horizon", [1, 2, 3, 4, 5], format_func=lambda x: f"{x} session" if x == 1 else f"{x} sessions")
    st.caption("The trade planner defaults to the next session; longer horizons show the path range through that session.")
    if st.button("↻  Refresh data & forecast", type="primary", use_container_width=True):
        try:
            run_scan(ledger)
            st.success("Forecast vintage updated.")
            st.rerun()
        except Exception as exc:
            st.error(f"Refresh failed: {type(exc).__name__}: {exc}")
    if report.get("run_at"):
        st.caption(f"Last run · {str(report['run_at']).replace('T', ' ')[:16]} UTC")
    st.markdown(f"**Status:** {report.get('status', 'WAITING FOR FIRST RUN')}")
    st.caption("Research estimates; no forecast guarantees a price or order fill.")

st.title("Natural Gas ETF · Trade Desk")
st.markdown("<p style='color:#91a2b9;margin-top:-.6rem'>UNG · Quantitative range forecast and buy-low / sell-high planning</p>", unsafe_allow_html=True)
st.markdown("<span class='desk-pill'>HistGradientBoosting range engine</span><span class='desk-pill'>DMA adaptive weighting</span><span class='desk-pill'>Online range calibration</span>", unsafe_allow_html=True)
desk_tab, ladder_tab, learning_tab, data_tab, sim_tab = st.tabs(["Trading desk", "Range ladder", "AI & learning", "Market data", "Simulation"])

with desk_tab:
    selected = consensus_for(forecasts, horizon)
    if selected is None:
        st.info("No current forecast is loaded. Configure the Alpaca keys, then use **Refresh data & forecast**.")
    else:
        ref, low, close, high = (float(selected[key]) for key in ("reference_close", "predicted_low", "expected_close", "predicted_high"))
        st.caption(f"Origin: {selected['origin_session']} · Target: {selected['target_session']} · Range state: {selected.get('calibration_status','WARMING_UP')}")
        previous=prior_consensus(ledger,selected)
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Reference close", money(ref), "Last completed RTH close")
        low_delta=f"{(low/ref-1)*100:+.2f}% vs close"+(f" · rev {low-float(previous['predicted_low']):+.2f}" if previous else "")
        close_delta=f"{(close/ref-1)*100:+.2f}% vs close"+(f" · rev {close-float(previous['expected_close']):+.2f}" if previous else "")
        high_delta=f"{(high/ref-1)*100:+.2f}% vs close"+(f" · rev {high-float(previous['predicted_high']):+.2f}" if previous else "")
        k2.metric("Forecast low", money(low), low_delta, delta_color="inverse")
        k3.metric("Expected close", money(close), close_delta)
        k4.metric("Forecast high", money(high), high_delta)
        target_records=[r for r in ledger.outside_rth_features() if r.get("target_session")==selected["target_session"]]
        if target_records:
            st.markdown("#### Outside-RTH monitor")
            mid=next((r for r in target_records if r.get("snapshot_slot")=="mid"),None)
            one_hour=next((r for r in target_records if r.get("snapshot_slot")=="one_hour"),None)
            post=next((r for r in reversed(target_records) if r.get("ung_after_hours_last_price") is not None),None)
            o1,o2,o3=st.columns(3)
            o1.metric("Premarket · 6:45 ET",money(mid.get("ung_premarket_mid_last_price")) if mid else "Not captured",f"{float(mid['ung_premarket_mid_return']):+.2%} vs prior close" if mid and mid.get("ung_premarket_mid_return") is not None else None)
            o2.metric("Premarket · 8:30 ET",money(one_hour.get("ung_premarket_1h_last_price")) if one_hour else "Not captured",f"{float(one_hour['ung_premarket_1h_return']):+.2%} vs prior close" if one_hour and one_hour.get("ung_premarket_1h_return") is not None else None)
            o3.metric("Prior after-hours close",money(post.get("ung_after_hours_last_price")) if post else "Not captured",f"{float(post['ung_after_hours_return']):+.2%} vs RTH close" if post and post.get("ung_after_hours_return") is not None else None)
        st.caption(f"Forecast range width: {(high-low)/ref*100:.2f}% of reference close.")
        with st.container(border=True):
            st.subheader("Build a limit-order plan")
            st.caption("Select levels inside the projected range. Neither level is a fill prediction.")
            entry_pct=st.slider("Buy level · low → expected close",0,100,20,5)
            exit_pct=st.slider("Sell target · expected close → high",0,100,75,5)
            shares=st.number_input("Shares per round trip",1,100000,100,10)
            costs=st.number_input("Estimated round-trip cost / share ($)",0.0,10.0,0.03,0.01,format="%.2f")
            buy=low+(close-low)*entry_pct/100
            sell=close+(high-close)*exit_pct/100
            net=(sell-buy-costs)*shares
            m1,m2,m3=st.columns(3)
            m1.metric("Buy limit",money(buy))
            m2.metric("Sell target",money(sell))
            m3.metric("Est. net / cycle",money(net),f"{(sell-buy-costs)/buy*100:+.2f}% / share")
            st.caption(f"Gross selected gap: {money(sell-buy)} per share. Your cost estimate may not include all fees, spread, slippage, or taxes.")
        st.subheader(f"Model confidence · {horizon}-session horizon")
        confidence=confidence_frame(scorecards,horizon)
        if not confidence.empty:
            st.dataframe(confidence,use_container_width=True,hide_index=True,column_config={
                "Historical confidence (%)":st.column_config.NumberColumn(format="%.1f%%"),
            })
            st.caption("Confidence is the model's observed past range coverage at this horizon. It is not a probability that today's forecast is correct or that an order fills.")
        else:
            st.info("Confidence percentages appear as forecasts mature. No completed outcomes are available yet.")

with ladder_tab:
    st.subheader("One-to-five-session range ladder")
    rows=[r for r in forecasts if r.get("model_name")=="dma_consensus"]
    if rows:
        ladder=pd.DataFrame([{"Horizon":f"{r['horizon']} session"+("" if r["horizon"]==1 else "s"),"Target date":r["target_session"],"Low":r["predicted_low"],"Expected close":r["expected_close"],"High":r["predicted_high"],"Width %":100*(r["predicted_high"]-r["predicted_low"])/r["reference_close"],"DMA":r.get("learning_status"),"Calibration":r.get("calibration_status")} for r in rows]).sort_values("Target date")
        st.dataframe(ladder,use_container_width=True,hide_index=True,column_config={"Low":st.column_config.NumberColumn(format="$%.2f"),"Expected close":st.column_config.NumberColumn(format="$%.2f"),"High":st.column_config.NumberColumn(format="$%.2f"),"Width %":st.column_config.NumberColumn(format="%.2f%%")})
        st.download_button("Download range ladder (CSV)",ladder.to_csv(index=False),"ung-forecast-ladder.csv","text/csv")
        st.caption("For horizons 2–5, low/high describe the expected price-path extremes from the current reference close through the target session.")
    else:
        st.info("Refresh the desk to create the first forecast ladder.")

with learning_tab:
    st.subheader("How the adaptive forecast works")
    st.markdown("The desk has **one base forecasting algorithm and one adaptive ensemble learner**. DMA is a meta-learning layer, not an unrelated second price model.")
    a,b=st.columns(2)
    with a:
        st.markdown("#### 1 · HistGradientBoosting range engine")
        st.write("Fits low, central, and high estimates from UNG history. It trains a full-feature model and feature-group experts for momentum, volatility, trend, volume, gas context, and premarket snapshots once enough snapshots exist.")
        st.caption("Low/high use 10th/90th quantile losses; the center uses squared-error loss. The engine refits from available history on each run.")
    with b:
        st.markdown("#### 2 · Dynamic Model Averaging (DMA)")
        st.write("DMA blends expert forecasts separately for each horizon. Once outcomes mature, experts that better forecast lows, closes, and highs receive more weight; weak experts lose weight gradually.")
        st.caption("Weights start equally and adapt after 20 matured outcomes per expert and horizon. A small floor lets a weak expert recover in a changed regime.")
    st.markdown("#### Range correction · Online adaptive calibration")
    st.write("A horizon-specific calibration layer checks whether actual lows and highs stayed inside earlier ranges. Repeated misses widen future ranges; sustained coverage can tighten them. Five-session forecasts update only after all five sessions mature.")
    st.warning("This is adaptive statistical learning, not a generative AI or a promise of accuracy. UNG can move beyond its forecast range; this page reports historical evidence and data gaps so you can assess the estimate.")
    selected=consensus_for(forecasts,horizon)
    if selected:
        st.markdown(f"#### Current expert blend · {horizon}-session horizon")
        weight_map=selected.get("component_weights",{})
        wf=pd.DataFrame([{"HGB feature expert":n.replace("hgb_expert_","").replace("hist_gradient_boosting","all features"),"DMA weight":w,"Matured samples":selected.get("learning_sample_count",0)} for n,w in weight_map.items()])
        if not wf.empty:
            wf=wf.sort_values("DMA weight",ascending=False)
            st.dataframe(wf,use_container_width=True,hide_index=True,column_config={"DMA weight":st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.1%%")})
            experts=[r for r in forecasts if r.get("model_name") != "dma_consensus" and int(r.get("horizon",0)) == horizon]
            if experts:
                simple_mean=sum(float(r["expected_close"]) for r in experts)/len(experts)
                attribution=pd.DataFrame([{
                    "Feature expert":r.get("model_name"),
                    "Model expected close":r["expected_close"],
                    "DMA weight":weight_map.get(r.get("model_name"),0.0),
                    "Weighted adjustment vs equal blend ($)":weight_map.get(r.get("model_name"),0.0)*(float(r["expected_close"])-simple_mean),
                } for r in experts]).sort_values("DMA weight",ascending=False)
                st.markdown("#### What moved the blend?")
                st.caption("This decomposes the DMA ensemble's difference from an equal-weight blend. It explains model influence, not causal price drivers.")
                st.dataframe(attribution,use_container_width=True,hide_index=True,column_config={
                    "Model expected close":st.column_config.NumberColumn(format="$%.2f"),
                    "Weighted adjustment vs equal blend ($)":st.column_config.NumberColumn(format="$%.3f"),
                })
    st.markdown("#### Model confidence percentages")
    confidence=confidence_frame(scorecards,horizon)
    if not confidence.empty:
        st.dataframe(confidence,use_container_width=True,hide_index=True,column_config={
            "Historical confidence (%)":st.column_config.NumberColumn(format="%.1f%%"),
        })
    else:
        st.info("No matured outcomes yet. Each completed session adds evidence for later updates.")
    latest_score=max((str(r.get("scored_at","")) for r in outcomes),default="none")
    st.caption(f"Recorded outcomes: {len(outcomes):,} · latest scoring update: {latest_score}")

with data_tab:
    st.subheader("Market inputs and availability")
    feeds=report.get("data_feeds",[])
    if feeds:
        ff=pd.DataFrame(feeds)
        cols=[c for c in ("source","status","latest_observation","latest_value","available_at","rows","role","note") if c in ff]
        st.dataframe(ff[cols],use_container_width=True,hide_index=True)
    else:
        st.info("Run a refresh to check configured sources.")
    st.markdown("#### Inputs in the forecast")
    st.write("**Connected:** UNG daily OHLCV, realized volatility/range, momentum, trend, volume, and collected premarket snapshots. **Not connected as forecast inputs:** NYMEX contract curve, weather vintages, LNG/production flows, options volatility, and full order-book depth.")
    group_descriptions={
        "momentum":"Recent returns and RSI",
        "volatility_range":"Rolling high-low ranges, realized volatility, and ATR",
        "trend":"Price relative to exponential moving averages",
        "volume":"Relative volume and volume z-score",
        "gas_context":"Henry Hub and weekly positioning/storage when available",
        "futures_curve":"Front-month and deferred natural-gas contracts (not connected)",
        "weather":"Demand forecasts and ensemble spread (not connected)",
        "fundamentals":"Storage surprises and positioning (partly connected)",
        "outside_rth_mid":"Prior after-hours and 6:45 a.m. premarket behavior",
        "outside_rth_1h":"Prior after-hours and 8:30 a.m. premarket behavior",
    }
    factor_rows = [{
        "Factor group": name.replace("_", " ").title(),
        "What it measures": description,
        "Feature fields": ", ".join(FORECAST_FEATURE_GROUPS.get(name, ())),
    } for name, description in group_descriptions.items()]
    st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)
    bars=st.session_state.get("latest_bars")
    if isinstance(bars,pd.DataFrame) and not bars.empty:
        st.markdown("#### Recent UNG sessions")
        st.dataframe(bars.tail(10).sort_index(ascending=False),use_container_width=True)
    st.markdown("#### Repository data footprint")
    used=storage_bytes()
    st.progress(min(1.0,used/DATA_BUDGET_BYTES),text=f"Tracked JSON data: {used/1024:.0f} KB of 8 MB working budget")
    st.caption("The scheduled collector uses a bounded rolling ledger, does not create monthly archive copies, and keeps large simulation results out of Git.")

with sim_tab:
    st.subheader("Walk-forward simulation")
    st.write("Replays forecasts across historical origins using only information available at each origin and matured outcomes for adaptive weights.")
    if st.button("Run 1–5 session simulation"):
        with st.spinner("Running historical simulation…"):
            try:
                fetched=fetch_alpaca_ung_daily()
                simulation=run_walk_forward_simulation(fetched.frame,outside_rth=outside_rth_frame(ledger.outside_rth_features()),min_train_rows=252,step=5,max_origins=126)
                simulation["price_source"]="alpaca"
                simulation["data_through"]=fetched.snapshot.published_through.isoformat()
                st.session_state["trade_desk_simulation"]=simulation
            except Exception as exc:
                st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
    simulation=st.session_state.get("trade_desk_simulation")
    if simulation:
        c1,c2,c3=st.columns(3)
        c1.metric("Forecast origins",simulation["simulation_origins"])
        c2.metric("Daily bars",simulation["bars"])
        c3.metric("Data through",simulation.get("data_through","—"))
        if simulation.get("metrics_by_horizon"):
            st.markdown("#### Walk-forward range confidence by horizon")
            metrics=pd.DataFrame(simulation["metrics_by_horizon"])
            if "range_coverage" in metrics:
                metrics=metrics.rename(columns={"range_coverage":"Historical confidence"})
                st.dataframe(metrics,use_container_width=True,hide_index=True,column_config={
                    "Historical confidence":st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.0f%%"),
                })
            else:
                st.dataframe(metrics,use_container_width=True,hide_index=True)
        st.download_button("Download simulation report (JSON)",json.dumps(simulation,indent=2),"ung-walk-forward-simulation.json","application/json")
    else:
        st.caption("Simulation output stays in this browser session and is not committed to GitHub.")

st.divider()
st.caption("Quantitative research and trade planning · Forecast ranges and estimated levels are uncertain and do not guarantee fills or profit.")
