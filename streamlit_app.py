"""Public read-only Streamlit dashboard for UNGMJ research."""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

import streamlit as st

BASE = (
    "https://raw.githubusercontent.com/marthandan77/UNGMJ/"
    "research-state/data/ledger/"
)
USER_AGENT = "UNGMJ-Research-Dashboard/0.2"
REPO_API = "https://api.github.com/repos/marthandan77/UNGMJ/contents/data/ledger/model-config.json"
MODEL_OPTIONS = {
    "ridge": "Ridge Regression",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "sgd_online": "SGD Online-Learning Challenger",
}


def save_model_config(enabled_models: list[str]) -> None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("Persistent settings are not enabled. Add a narrowly scoped GITHUB_TOKEN or GH_TOKEN secret to the Render service.")
    headers = {
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": USER_AGENT,
    }
    request = urllib.request.Request(REPO_API, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            current = json.loads(response.read().decode("utf-8"))
            sha = current.get("sha")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            sha = None
        else:
            raise RuntimeError("Could not read persisted settings from GitHub.") from exc
    config = {
        "schema_version": 1,
        "enabled_models": sorted(set(enabled_models)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "note": "User-controlled shadow model processing; historical forecasts are retained.",
    }
    body = {"message": "Update UNGMJ model processing settings", "content": base64.b64encode((json.dumps(config, indent=2, sort_keys=True) + "\\n").encode()).decode(), "branch": "research-state"}
    if sha:
        body["sha"] = sha
    request = urllib.request.Request(REPO_API, data=json.dumps(body).encode(), headers={**headers, "Content-Type": "application/json"}, method="PUT")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status not in (200, 201):
                raise RuntimeError("GitHub did not confirm the settings update.")
    except urllib.error.HTTPError as exc:
        raise RuntimeError("GitHub rejected the settings update. Check the token's repository Contents: write permission.") from exc


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

st.subheader("Model processing settings")
st.caption("These controls govern the next scheduled research run. Turning a model off does not delete its forecast history or matured scores.")
try:
    config = load_json("model-config.json")
except Exception:
    config = {"enabled_models": list(MODEL_OPTIONS)}
enabled_now = [key for key in config.get("enabled_models", list(MODEL_OPTIONS)) if key in MODEL_OPTIONS]
with st.form("model_settings_form"):
    enabled_selected = st.multiselect(
        "Models enabled for the next run",
        options=list(MODEL_OPTIONS),
        default=enabled_now,
        format_func=lambda key: MODEL_OPTIONS[key],
        help="Every enabled model issues its own forecast. Models are not averaged together.",
    )
    save_settings = st.form_submit_button("Save model settings")
if save_settings:
    try:
        save_model_config(enabled_selected)
        load_json.clear()
        st.success("Settings saved to the research-state ledger. They apply on the next scheduled run.")
        st.rerun()
    except Exception as exc:
        st.error(str(exc))
        st.warning("Settings were not saved; the processing configuration has not changed.")
if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")):
    st.info("Settings are read-only until a repository-scoped GITHUB_TOKEN or GH_TOKEN secret with Contents: write access is configured on the Render service.")

st.subheader("Model competition")
scored_rows = []
for outcome in outcomes:
    if outcome.get("model_name") and outcome.get("absolute_return_error") is not None:
        scored_rows.append(outcome)
if scored_rows:
    grouped = {}
    for row in scored_rows:
        key = (str(row.get("model_name", "unknown")), int(row.get("horizon_sessions", 0)))
        grouped.setdefault(key, []).append(row)
    competition = []
    for (model_key, horizon), rows in sorted(grouped.items()):
        errors = [float(row["absolute_return_error"]) for row in rows if row.get("absolute_return_error") is not None]
        briers = [float(row["brier_score"]) for row in rows if row.get("brier_score") is not None]
        directions = [bool(row["direction_correct"]) for row in rows if row.get("direction_correct") is not None]
        competition.append({
            "Model": MODEL_OPTIONS.get(model_key, model_key),
            "Horizon (sessions)": horizon,
            "Matured forecasts": len(rows),
            "MAE (return)": sum(errors) / len(errors) if errors else None,
            "Direction accuracy": sum(directions) / len(directions) if directions else None,
            "Brier score (lower is better)": sum(briers) / len(briers) if briers else None,
            "Evidence": "Early sample" if len(rows) < 30 else ("Developing" if len(rows) < 100 else "Larger sample"),
        })
    st.dataframe(competition, use_container_width=True, hide_index=True)
    st.caption("Compare models within the same horizon. MAE and Brier score are lower-is-better; directional accuracy is not sufficient by itself. Small samples are noisy.")
else:
    st.info("Model competition scores will appear after forecast horizons mature. Existing forecasts are preserved for audit and later scoring.")

st.subheader("Forecast history")
st.caption(
    "A record of the predictions the system made. We keep older predictions so we can "
    "check later how accurate they were. These are estimates, not promises."
)
if visible_forecasts:
    ordered = list(reversed(visible_forecasts[-100:]))
    st.dataframe(
        [
            {
                "Made on (UTC)": row.get("as_of"),
                "Starting date": row.get("origin_session"),
                "Looking ahead": f"{row.get('horizon_sessions')} trading day(s)",
                "UNG price at start": row.get("reference_close"),
                "Estimated future price": row.get("expected_close"),
                "Chance price rises": row.get("probability_up"),
                "Possible lower price": row.get("predicted_low"),
                "Possible higher price": row.get("predicted_high"),
                "Past examples used": row.get("calibration_sample_count"),
                "How chance was estimated": row.get("calibration_method"),
                "Record status": "Cancelled — not scored" if row.get("status") == "VOID" else row.get("status", "Not specified"),
                "Why cancelled?": row.get("invalidation_reason", ""),
            }
            for row in ordered
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No predictions have been recorded yet.")

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
