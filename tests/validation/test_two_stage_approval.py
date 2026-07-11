from __future__ import annotations

from ung_forecast.shadow import OperatingMode, ShadowReadiness
from ung_forecast.validation.approval import (
    ApprovalDecision,
    StatisticalApprovalCriteria,
    TradingApprovalCriteria,
    ValidationMetrics,
    evaluate_statistical_approval,
    evaluate_trading_approval,
)


def test_statistical_approval_does_not_require_economic_value() -> None:
    model = ValidationMetrics(0.50, 0.90, 0.05, 500)
    baseline = ValidationMetrics(0.70, 1.10, 0.07, 500)
    decision = evaluate_statistical_approval(
        model,
        baseline,
        StatisticalApprovalCriteria(minimum_samples=250),
    )
    assert decision.approved is True


def test_trading_approval_requires_execution_model_and_shadow_samples() -> None:
    decision = evaluate_trading_approval(
        statistical_decision=ApprovalDecision(True, ()),
        economic_value=0.02,
        shadow_samples=20,
        execution_model_configured=False,
        criteria=TradingApprovalCriteria(minimum_shadow_samples=100),
    )
    assert decision.approved is False
    assert "execution_model_not_configured" in decision.reasons
    assert "insufficient_shadow_samples" in decision.reasons


def test_shadow_mode_never_enables_order_execution() -> None:
    readiness = ShadowReadiness(
        mode=OperatingMode.SHADOW,
        statistical_approval=ApprovalDecision(True, ()),
        trading_approval=ApprovalDecision(True, ()),
        scored_forecasts=500,
    )
    assert readiness.may_present_actionable_advice is False
    assert readiness.may_execute_orders is False
