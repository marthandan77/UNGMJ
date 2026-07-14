"""Deterministic nightly research report from ledger evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ung_forecast.persistence.schemas import ForecastRecord, ForecastRecordStatus


@dataclass(frozen=True, slots=True)
class NightlyResearchReport:
    report_date: date
    total_records: int
    pending_records: int
    scored_records: int
    invalid_records: int
    directional_accuracy: float | None
    average_confidence: float | None
    stale_or_invalid_data_records: int
    model_versions: tuple[str, ...]


def build_nightly_research_report(
    records: list[ForecastRecord],
    *,
    report_date: date,
) -> NightlyResearchReport:
    selected = [record for record in records if record.created_at.date() == report_date]
    scored = [record for record in selected if record.status is ForecastRecordStatus.SCORED]
    pending = [record for record in selected if record.status is ForecastRecordStatus.PENDING]
    invalid = [record for record in selected if record.status is ForecastRecordStatus.INVALID]

    correct = 0
    eligible = 0
    for record in scored:
        if record.outcome is None:
            continue
        probabilities = record.forecast.probabilities
        predicted = max(
            {
                "LOWER_FIRST": probabilities.lower_first,
                "UPPER_FIRST": probabilities.upper_first,
                "NEITHER": probabilities.neither,
            },
            key=lambda key: {
                "LOWER_FIRST": probabilities.lower_first,
                "UPPER_FIRST": probabilities.upper_first,
                "NEITHER": probabilities.neither,
            }[key],
        )
        eligible += 1
        if predicted == record.outcome.outcome.value:
            correct += 1

    confidence_values = [record.forecast.confidence_score for record in selected]
    stale_or_invalid = sum(
        record.forecast.data_freshness.value != "CURRENT" for record in selected
    )
    return NightlyResearchReport(
        report_date=report_date,
        total_records=len(selected),
        pending_records=len(pending),
        scored_records=len(scored),
        invalid_records=len(invalid),
        directional_accuracy=(correct / eligible if eligible else None),
        average_confidence=(
            sum(confidence_values) / len(confidence_values) if confidence_values else None
        ),
        stale_or_invalid_data_records=stale_or_invalid,
        model_versions=tuple(
            sorted({record.forecast.identity.model_version for record in selected})
        ),
    )
