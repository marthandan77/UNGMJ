"""Command-line entry point for a fail-closed daily research run."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .contracts import DataValidationError, ForecastVintage
from .features import build_features, build_forward_targets
from .evaluation import evaluate_strategies_and_features, summarize_evaluation
from .feeds import fetch_eia_henry_hub_spot, fetch_ung_sources, latest_session_agreement
from .ledger import ForecastLedger
from .model import fit_and_forecast, walk_forward_predictions

LOG = logging.getLogger("ungmj")


def run_once(output_dir: str = "data/ledger") -> dict[str, object]:
    now = datetime.now(timezone.utc)
    try:
        first, second = fetch_ung_sources()
        agreed, agreement_note = latest_session_agreement(first, second)
        if not agreed:
            raise DataValidationError(agreement_note)
        gas, gas_snapshot, _ = fetch_eia_henry_hub_spot()
        ung = first.frame
        # Henry Hub is explanatory only and conservatively lagged by feature builder.
        features = build_features(ung, gas)
        latest_features = features.iloc[[-1]]
        if latest_features.isna().all(axis=None):
            raise DataValidationError("Latest feature row is entirely missing")
        ledger = ForecastLedger(output_dir)
        emitted = 0
        for horizon in (1, 2, 3, 5):
            targets = build_forward_targets(ung, horizon)
            # The training label is truncated before the current feature row.
            # A label is eligible only if its entire forward horizon ended before the latest session.
            last_eligible = ung.index[-(horizon + 1)]
            train_targets = targets.loc[targets.index <= last_eligible]
            oof = walk_forward_predictions(
                features, targets, min_train_rows=250, step=10, horizon_sessions=horizon
            )
            calibration_residuals = (
                oof["actual_return"].to_numpy(dtype=float) - oof["predicted_return"].to_numpy(dtype=float)
                if len(oof) >= 50 else None
            )
            forecast = fit_and_forecast(
                features,
                train_targets,
                float(ung.iloc[-1]["close"]),
                min_train_rows=100,
                calibration_residuals=calibration_residuals,
            )
            fingerprint = hashlib.sha256(
                (first.snapshot.content_sha256 + second.snapshot.content_sha256 + gas_snapshot.content_sha256
                 + str(latest_features.index[0]) + str(horizon)).encode()
            ).hexdigest()
            vintage = ForecastVintage(
                forecast_id=f"UNG-{latest_features.index[0].date().isoformat()}T{now.strftime('%H%M%S')}-{horizon}D-{fingerprint[:12]}",
                instrument="UNG",
                as_of=now,
                origin_session=latest_features.index[0].date(),
                horizon_sessions=horizon,
                reference_close=float(ung.iloc[-1]["close"]),
                probability_up=forecast.probability_up,
                expected_return=forecast.expected_return,
                expected_close=forecast.expected_close,
                predicted_low=forecast.predicted_low,
                predicted_high=forecast.predicted_high,
                lower_return_quantile=forecast.lower_return_quantile,
                upper_return_quantile=forecast.upper_return_quantile,
                calibration_sample_count=forecast.calibration_sample_count,
                calibration_method=forecast.calibration_method,
                model_name=forecast.model_name,
                model_version=forecast.model_version,
                input_fingerprint=fingerprint,
                status="PROVISIONAL",
            )
            ledger.append_forecast(vintage)
            emitted += 1
        scored = ledger.score_matured(ung, now=now)
        evaluation_report: dict[str, object] = {"status": "BLOCKED", "reason": "Not yet evaluated"}
        try:
            evaluation_rows = []
            for eval_horizon in (1, 2, 3, 5):
                eval_targets = build_forward_targets(ung, eval_horizon)
                horizon_rows = evaluate_strategies_and_features(
                    features, eval_targets, horizon_sessions=eval_horizon,
                    min_train_rows=250, step=20,
                )
                evaluation_rows.extend(horizon_rows.to_dict(orient="records"))
            evaluation_frame = pd.DataFrame(evaluation_rows)
            evaluation_report = {
                "status": "COMPLETE",
                "method": "chronological walk-forward with purged horizon labels",
                "summaries": summarize_evaluation(evaluation_frame),
                "automatic_promotion": False,
            }
        except Exception as eval_exc:
            evaluation_report = {
                "status": "BLOCKED",
                "error_type": type(eval_exc).__name__,
                "error": str(eval_exc)[:500],
                "automatic_promotion": False,
            }
        Path(output_dir, "evaluation.json").write_text(
            json.dumps(evaluation_report, indent=2, sort_keys=True) + "\\n"
        )
        report = {
            "run_at": now.isoformat(),
            "status": "PROVISIONAL",
            "latest_ung_session": str(ung.index[-1].date()),
            "latest_henry_hub_session": str(gas.index[-1].date()),
            "provider_agreement": agreement_note,
            "forecasts_emitted": emitted,
            "outcomes_scored": scored,
            "evaluation": evaluation_report,
            "source_providers": [first.snapshot.provider, second.snapshot.provider, gas_snapshot.provider],
            "uncertainty_status": "Return intervals use purged walk-forward residuals when at least 50 are available; high-low range calibration and coverage review remain outstanding.",
            "note": "Research output only; not investment advice or a trading instruction.",
        }
    except Exception as exc:
        LOG.exception("Run blocked")
        report = {
            "run_at": now.isoformat(),
            "status": "BLOCKED",
            "forecasts_emitted": 0,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "latest-run.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="UNG research forecast pipeline")
    parser.add_argument("--output-dir", default="data/ledger")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    report = run_once(args.output_dir)
    if report.get("status") == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
