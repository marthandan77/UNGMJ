# Requirement Traceability Matrix

Status values: `PENDING`, `IMPLEMENTED`, `TESTED`, `BLOCKED`.

| Requirement | Planned implementation | Required test | Status |
|---|---|---|---|
| AC-DATA-001 | `src/data/validator.py` | `tests/data/test_completed_bars.py` | PENDING |
| AC-DATA-002 | `src/data/validator.py` | `tests/data/test_duplicate_timestamps.py` | PENDING |
| AC-DATA-003 | `src/data/validator.py` | `tests/data/test_stale_data.py` | PENDING |
| AC-DATA-004 | `src/data/calendar.py` | `tests/data/test_timezone_policy.py` | PENDING |
| AC-DATA-005 | `src/data/schemas.py` | `tests/data/test_provenance.py` | PENDING |
| AC-DATA-006 | `src/data/adjustments.py` | `tests/data/test_adjustment_policy.py` | PENDING |
| AC-DATA-007 | `src/models/registry.py` | `tests/models/test_ng_fallback.py` | PENDING |
| AC-HOR-001..005 | `src/ung_forecast/horizons.py` | `tests/test_foundation.py` | IMPLEMENTED |
| AC-HOR-006 | `src/models/registry.py` | `tests/models/test_independent_horizons.py` | PENDING |
| AC-FEAT-001 | `src/features/library.py` | `tests/features/test_training_live_parity.py` | PENDING |
| AC-FEAT-002 | `src/features/audit.py` | `tests/features/test_no_future_sources.py` | PENDING |
| AC-FEAT-003 | `src/features/library.py` | `tests/features/test_no_centered_windows.py` | PENDING |
| AC-LABEL-001 | `src/labels/triple_barrier.py` | `tests/labels/test_path_outcomes.py` | PENDING |
| AC-LABEL-002 | `src/barriers/engine.py` | `tests/labels/test_barrier_parity.py` | PENDING |
| AC-MODEL-001 | `src/ung_forecast/schemas.py` | `tests/test_foundation.py` | IMPLEMENTED |
| AC-MODEL-002 | `src/models/elastic_net.py` | `tests/models/test_baseline_model.py` | PENDING |
| AC-VAL-001..004 | `src/validation/walk_forward.py` | `tests/validation/test_no_leakage.py` | PENDING |
| AC-VAL-005..006 | `src/validation/approval.py` | `tests/validation/test_approval_gate.py` | PENDING |
| AC-DEC-001..004 | `src/decision/expected_value.py` | `tests/decision/test_expected_value_advice.py` | PENDING |
| AC-CONF-001 | `src/confidence/engine.py` | `tests/confidence/test_confidence_formula.py` | PENDING |
| AC-EXP-001..003 | `src/explanations/renderer.py` | `tests/explanations/test_traceable_templates.py` | PENDING |
| AC-CROSS-001..002 | `src/coordinator/cross_horizon.py` | `tests/coordinator/test_no_probability_mutation.py` | PENDING |
| AC-LEARN-001..004 | `src/learning/champion_challenger.py` | `tests/learning/test_promotion_controls.py` | PENDING |
| AC-UI-001..004 | `dashboard/` | `tests/dashboard/test_render_contract.py` | PENDING |
| AC-DEL-001..004 | `tools/compliance_report.py` | `tests/compliance/test_traceability.py` | PENDING |

This matrix must be updated in the same pull request as each implementation. A requirement is complete only when implementation and its mapped automated test are present and passing.
