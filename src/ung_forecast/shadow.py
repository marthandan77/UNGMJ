"""Shadow-mode controls for non-executing live observation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ung_forecast.validation.approval import ApprovalDecision


class OperatingMode(StrEnum):
    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"
    LIVE_DECISION_SUPPORT = "LIVE_DECISION_SUPPORT"


@dataclass(frozen=True, slots=True)
class ShadowReadiness:
    mode: OperatingMode
    statistical_approval: ApprovalDecision
    trading_approval: ApprovalDecision
    scored_forecasts: int

    @property
    def may_present_actionable_advice(self) -> bool:
        return (
            self.mode is OperatingMode.LIVE_DECISION_SUPPORT
            and self.statistical_approval.approved
            and self.trading_approval.approved
        )

    @property
    def may_execute_orders(self) -> bool:
        return False
