# Requirement Traceability Matrix

Status values: `PENDING`, `IMPLEMENTED`, `TESTED`, `BLOCKED`.

| Requirement | Planned implementation | Required test | Status |
|---|---|---|---|
| AC-DATA-001 | `src/ung_forecast/data/validator.py` | `tests/data/test_validation.py` | IMPLEMENTED |
| AC-DATA-002 | `src/ung_forecast/data/validator.py` | `tests/data/test_validation.py` | IMPLEMENTED |
| AC-DATA-003 | `src/ung_forecast/data/validator.py` | `tests/data/test_validation.py` | IMPLEMENTED |
| AC-DATA-004 | `src/ung_forecast/data/validator.py` | `tests/data/test_validation.py` | IMPLEMENTED |
| AC-DATA-005 | `src/ung_forecast/data/schemas.py` | `tests/data/test_cache.py` | IMPLEMENTED |
| AC-DATA-006 | `src/ung_forecast/configuration.py`, `src/ung_forecast/data/provider.py` | `tests/data/test_provider.py` | IMPLEMENTED |
| AC-DATA-007 | `src/ung_forecast/models/registry.py` | `tests/models/test_models.py` | IMPLEMENTED |
| AC-HOR-001..005 | `src/ung_forecast/horizons.py` | `tests/test_foundation.py` | IMPLEMENTED |
| AC-HOR-006 | `src/ung_forecast/models/registry.py` | `tests/models/test_models.py` | IMPLEMENTED |
| AC-FEAT-001 | `src/ung_forecast/features/formulas.py`, `src/ung_forecast/features/engine.py` | `tests/features/test_engine.py` | IMPLEMENTED |
| AC-FEAT-002 | `src/ung_forecast/features/engine.py` | `tests/features/test_engine.py` | IMPLEMENTED |
| AC-FEAT-003 | `src/ung_forecast/features/formulas.py` | `tests/features/test_formulas.py`, `tests/features/test_engine.py` | IMPLEMENTED |
| AC-LABEL-001 | `src/ung_forecast/labels/triple_barrier.py` | `tests/labels/test_barriers_and_labels.py` | IMPLEMENTED |
| AC-LABEL-002 | `src/ung_forecast/barriers/engine.py`, `src/ung_forecast/labels/triple_barrier.py` | `tests/labels/test_barriers_and_labels.py` | IMPLEMENTED |
| AC-MODEL-001 | `src/ung_forecast/schemas.py`, `src/ung_forecast/models/elastic_net.py` | `tests/test_foundation.py`, `tests/models/test_models.py` | IMPLEMENTED |
| AC-MODEL-002 | `src/ung_forecast/models/elastic_net.py` | `tests/models/test_models.py` | IMPLEMENTED |
| AC-VAL-001..004 | `src/ung_forecast/validation/walk_forward.py` | `tests/validation/test_walk_forward.py` | IMPLEMENTED |
| AC-VAL-005..006 | `src/validation/approval.py` | `tests/validation/test_approval_gate.py` | PENDING |
| AC-DEC-001..004 | `src/ung_forecast/decision/expected_value.py` | `tests/decision/test_expected_value_and_confidence.py` | IMPLEMENTED |
| AC-CONF-001 | `src/ung_forecast/confidence/engine.py` | `tests/decision/test_expected_value_and_confidence.py` | IMPLEMENTED |
| AC-EXP-001..003 | `src/explanations/renderer.py` | `tests/explanations/test_traceable_templates.py` | PENDING |
| AC-CROSS-001..002 | `src/coordinator/cross_horizon.py` | `tests/coordinator/test_no_probability_mutation.py` | PENDING |
| AC-LEARN-001..004 | `src/learning/champion_challenger.py` | `tests/learning/test_promotion_controls.py` | PENDING |
| AC-UI-001..004 | `dashboard/` | `tests/dashboard/test_render_contract.py` | PENDING |
| AC-DEL-001..004 | `tools/compliance_report.py` | `tests/compliance/test_traceability.py` | PENDING |

This matrix must be updated in the same pull request as each implementation. A requirement is complete only when implementation and its mapped automated test are present and passing.
