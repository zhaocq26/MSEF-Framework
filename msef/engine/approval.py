"""Human approval layered on top of quality gates.

Two approval types:
1. Stage-level: triggered between stages, judges whether the stage output is
   deliverable. Risk score combines change scope, critical-module touches, and
   high-risk item ratio:
       r_stage = w1·(changed files / total files) + w2·critical touch + w3·(high-risk items / total risk items)
2. Operation-level: triggered before high-risk operations inside a stage,
   judges whether the operation is safe to execute. Risk score combines
   operation type, impact scope, and reversibility:
       r_op = w1·type_weight + w2·(impact scope / total scope) + w3·(1 - reversibility)

Flow: submit → respond → decide → execute → record (timeout without response
pauses the task and escalates).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from msef.config import ApprovalConfig, OP_RISK_LEVELS
from msef.types import (
    ApprovalMode,
    ApprovalRequest,
    ApprovalVerdict,
    Operation,
    OperationRiskLevel,
    Stage,
    StageOutput,
)

logger = logging.getLogger(__name__)

# Approvers per stage
APPROVERS_BY_STAGE: Dict[str, str] = {
    "plan": "technical lead",
    "code": "module owner",
    "test": "test lead",
    "review": "code review lead",
    "deliver": "project lead",
}


# ---------------------------------------------------------------------------
# Risk evaluators
# ---------------------------------------------------------------------------
class StageRiskEvaluator:
    """Stage-level risk evaluation."""

    def __init__(self, weights: Optional[List[float]] = None,
                 critical_modules: Optional[List[str]] = None,
                 project_total_files: int = 100) -> None:
        self.w = weights or [0.3, 0.4, 0.3]
        self.critical_modules = set(critical_modules or [])
        self.project_total_files = max(1, project_total_files)

    def evaluate(self, output: StageOutput, risk_items: List[Dict[str, Any]],
                 touches_critical: bool = False) -> float:
        """Compute the stage output risk score r_stage ∈ [0,1]."""
        n_files = max(0, len(output.files))
        scale = n_files / self.project_total_files
        high = sum(
            1 for item in risk_items
            if item.get("level", "low") in ("high", "extreme")
        )
        total = max(1, len(risk_items))
        return min(1.0, self.w[0] * scale + self.w[1] * float(touches_critical) + self.w[2] * (high / total))

    def triggers(self, risk: float, threshold: float) -> bool:
        return risk >= threshold


class OperationRiskEvaluator:
    """Operation-level risk evaluation."""

    def __init__(self, weights: Optional[List[float]] = None,
                 operation_type_weights: Optional[Dict[str, float]] = None,
                 project_total_scope: int = 100) -> None:
        self.w = weights or [0.5, 0.3, 0.2]
        self.type_weights = operation_type_weights or {
            name: meta["weight"] for name, meta in OP_RISK_LEVELS.items()
        }
        self.project_total_scope = max(1, project_total_scope)

    def evaluate(self, op: Operation) -> float:
        """Compute the operation risk score r_op ∈ [0,1]."""
        type_w = self.type_weights.get(op.risk_level.value, 0.5)
        scope = min(1.0, op.impact_scope / self.project_total_scope)
        irrev = 1.0 - max(0.0, min(1.0, op.reversibility))
        return min(1.0, self.w[0] * type_w + self.w[1] * scope + self.w[2] * irrev)

    def triggers(self, risk: float, threshold: float) -> bool:
        return risk >= threshold


# ---------------------------------------------------------------------------
# Approval service
# ---------------------------------------------------------------------------
class HumanApprover:
    """Human approval actor (swappable with a real approval system/bot)."""

    def approve(self, request: ApprovalRequest) -> ApprovalResult:
        """Approve the request (default: approve). Subclasses may override with rules."""
        return ApprovalResult(request.id, ApprovalVerdict.APPROVED, "Approved by policy")


@dataclass
class ApprovalResult:
    """Approval result."""

    request_id: str
    verdict: ApprovalVerdict
    comment: str = ""
    decided_at: float = field(default_factory=time.time)


class ApprovalService:
    """Human approval service: evaluate → trigger → approve → execute/reject → record."""

    def __init__(self, config: Optional[ApprovalConfig] = None,
                 approver: Optional[HumanApprover] = None,
                 critical_modules: Optional[List[str]] = None,
                 project_total_files: int = 100,
                 project_total_scope: int = 100,
                 on_approval_record: Optional[Callable[[ApprovalRequest], None]] = None) -> None:
        self.config = config or ApprovalConfig()
        self.approver = approver or HumanApprover()
        self.stage_risk = StageRiskEvaluator(
            weights=config.stage_risk_weights if config else None,
            critical_modules=critical_modules,
            project_total_files=project_total_files,
        )
        self.op_risk = OperationRiskEvaluator(
            weights=config.op_risk_weights if config else None,
            project_total_scope=project_total_scope,
        )
        # Approval-record callback (persists records to the memory system)
        self.on_approval_record = on_approval_record
        self.requests: List[ApprovalRequest] = []
        self.human_interventions = 0

    # ------------------------------------------------------------------
    def should_stage_approve(self, stage: Stage) -> bool:
        """Stage-level approval trigger: review/deliver always, others per config."""
        return stage.value in self.config.stage_approval_stages

    def evaluate_stage(self, output: StageOutput, risk_items: List[Dict[str, Any]],
                       touches_critical: bool = False) -> float:
        return self.stage_risk.evaluate(output, risk_items, touches_critical)

    def check_stage_approval(self, stage: Stage, output: StageOutput,
                             risk_items: List[Dict[str, Any]],
                             touches_critical: bool = False) -> Optional[ApprovalRequest]:
        """Stage-level approval: decide whether to trigger."""
        risk = self.evaluate_stage(output, risk_items, touches_critical)
        if not self.should_stage_approve(stage):
            return None
        if not self.op_risk.triggers(risk, self.config.risk_threshold):
            return None
        request = ApprovalRequest.create(
            ApprovalMode.STAGE, stage,
            payload={"output": output.to_dict(), "risk_items": risk_items},
            risk_assessment={
                "score": risk,
                "threshold": self.config.risk_threshold,
                "touches_critical": touches_critical,
            },
            approver=APPROVERS_BY_STAGE.get(stage.value, "approval owner"),
        )
        self._submit(request)
        return request

    def check_operation_approval(self, op: Operation) -> Optional[ApprovalRequest]:
        """Operation-level approval: decide whether a high-risk operation triggers."""
        risk = self.op_risk.evaluate(op)
        if not self.op_risk.triggers(risk, self.config.risk_threshold):
            return None
        request = ApprovalRequest.create(
            ApprovalMode.OPERATION, op.stage,
            payload={"operation": {"id": op.id, "name": op.name, "description": op.description}},
            risk_assessment={
                "score": risk,
                "threshold": self.config.risk_threshold,
                "op_type": op.risk_level.value,
                "impact_scope": op.impact_scope,
                "reversibility": op.reversibility,
            },
            approver=APPROVERS_BY_STAGE.get(op.stage.value, "approval owner"),
        )
        self._submit(request)
        return request

    def _submit(self, request: ApprovalRequest) -> None:
        """Submit: pause execution and create the approval request."""
        self.requests.append(request)
        logger.info(
            "[approval-submit] %s %s stage, risk=%.2f, approver=%s",
            request.mode.value, request.stage.value,
            request.risk_assessment.get("score", 0.0), request.approver,
        )

    # ------------------------------------------------------------------
    def resolve(self, request: ApprovalRequest, verdict: Optional[ApprovalVerdict] = None,
                comment: str = "") -> ApprovalResult:
        """Respond + decide + record."""
        self.human_interventions += 1
        result = self.approver.approve(request) if verdict is None else ApprovalResult(
            request.id, verdict, comment
        )
        request.result = result.verdict
        request.comment = result.comment
        request.resolved_at = time.time()
        logger.info(
            "[approval-decision] %s -> %s (%s)",
            request.id, result.verdict.value, request.comment,
        )
        if self.config.record_to_memory and self.on_approval_record:
            self.on_approval_record(request)   # persist approval records to the memory system
        return result

    def decide(self, request: ApprovalRequest) -> ApprovalResult:
        """Simplified entry point: approve per current config."""
        return self.resolve(request)
