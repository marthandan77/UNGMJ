"""Public read-only Streamlit dashboard for UNGMJ research."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import streamlit as st

BASE = (
    "https://raw.githubusercontent.com/marthandan77/UNGMJ/"
    "research-state/data/ledger/"
)
USER_AGENT = "UNGMJ-Research-Dashboard/0.1"


def fetch_text(path: str) -> str:
    request = urllib.request.Request(
        BASE + path,
        headers={"User-Agent": USER_AGENT, "Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError(f"Could not load {path}: HTTP {response.status}")
        return response.read().decode("utf-8")


@st.cache_data(ttl=300, show_spinner=False)
def load_json(path: str) -> dict[str, Any]:
    raw = fetch_text(path).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate a literal backslash-n left by an older pipeline run.
        if raw.endswith("\\n"):
            return json.loads(raw[:-2])
        raise


@st.cache_data(ttl=300, show_spinner=False)
def load_jsonl(path: str) -> list[dict[str, Any]]:
    try:
        raw = fetch_text(path)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise
    return [json.loads(line) for line in raw.splitlines() if line.strip()]



st.set_page_config(page_title="UNGMJ Research", page_icon="📈", layout="wide")

st.title("UNGMJ — UNG ETF research forecasts")
st.caption(
    "Probabilistic research estimates for 1, 2, 3, and 5 trading sessions. "
    "Not investment advice or trade instructions."
)
if st.button("Refresh data"):
    load_json.clear()
    load_jsonl.clear()
    st.rerun()

try:
    report = load_json("latest-run.json")
    forecasts = load_jsonl("forecasts.jsonl")
    outcomes = load_jsonl("outcomes.jsonl")
    invalidations = load_jsonl("invalidations.jsonl")
    evaluation = load_json("evaluation.json")
except Exception as exc:
    st.error(f"Research data unavailable: {type(exc).__name__}: {exc}")
    st.stop()

invalidated = {
    str(item["forecast_id"]): str(item.get("reason", "Invalidated"))
    for item in invalidations
}
visible_forecasts = []
for item in forecasts:
    row = dict(item)
    if str(item.get("forecast_id")) in invalidated:
        row["status"] = "VOID"
        row["invalidation_reason"] = invalidated[str(item["forecast_id"])]
    visible_forecasts.append(row)

status = str(report.get("status", "UNKNOWN"))
st.subheader(f"Run status: {status}")
st.write(
    f"Run at: {report.get('run_at', 'unknown')} UTC  |  "
    f"UNG session: {report.get('latest_ung_session', '—')}  |  "
    f"Henry Hub session: {report.get('latest_henry_hub_session', '—')}"
)
if status != "READY":
    st.warning(
        "Forecasts are provisional or blocked. Independent UNG source confirmation, "
        "data quality, and uncertainty calibration govern release status."
    )
if report.get("error"):
    st.error(str(report["error"]))
if report.get("run_validity_note"):
    st.warning(str(report["run_validity_note"]))

m1, m2, m3, m4 = st.columns(4)
m1.metric("UNG history rows", str(report.get("ung_history_rows", "—")))
m2.metric("Henry Hub rows", str(report.get("henry_hub_history_rows", "—")))
m3.metric("Valid forecasts this run", str(report.get("valid_forecasts_emitted", report.get("forecasts_emitted", 0))))
m4.metric("Matured outcomes", str(len(outcomes)))

st.subheader("Latest forecast vintages")
if visible_forecasts:
    ordered = list(reversed(visible_forecasts[-100:]))
    st.dataframe(
        [
            {
                "Issued (UTC)": row.get("as_of"),
                "Origin session": row.get("origin_session"),
                "Horizon": f"{row.get('horizon_sessions')} sessions",
                "Reference close": row.get("reference_close"),
                "Expected close": row.get("expected_close"),
                "P(up)": row.get("probability_up"),
                "Predicted low": row.get("predicted_low"),
                "Predicted high": row.get("predicted_high"),
                "Calibration N": row.get("calibration_sample_count"),
                "Calibration method": row.get("calibration_method"),
                "Status": row.get("status"),
                "Audit reason": row.get("invalidation_reason", ""),
            }
            for row in ordered
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No forecast vintages are available yet.")

st.subheader("Walk-forward evaluation")
st.write(
    f"Evaluation status: {evaluation.get('status', 'NOT RUN')}  |  "
    f"Automatic model promotion: {evaluation.get('automatic_promotion', False)}"
)
summaries = report.get("evaluation", {}).get("summaries") or evaluation.get("summaries") or []
if summaries:
    st.dataframe(summaries, use_container_width=True, hide_index=True)
else:
    st.info("No evaluation summary is available in the latest report.")

st.subheader("Matured outcome scoring")
if outcomes:
    st.dataframe(list(reversed(outcomes[-100:])), use_container_width=True, hide_index=True)
else:
    st.info("No forecast horizon has matured yet. Outcomes are scored once after maturity.")

st.subheader("Data and governance")
st.write(
    "Target: UNG ETF. Henry Hub daily spot is explanatory only. Weather, paid data feeds, "
    "broker integration, order execution, and email are excluded. Ridge remains the fixed "
    "baseline; challengers stay shadow-only. All evaluation is chronological and purged. "
    "Invalidated validation vintages remain in the audit ledger but are excluded from scoring."
)
