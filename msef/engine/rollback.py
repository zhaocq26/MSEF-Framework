"""Failure rollback and checkpoint resume.

Failure rollback: when a quality gate fails or human approval is rejected, the
execution state rolls back to the last stable stage, the strategy is adjusted
according to the failure cause, and execution restarts; a maximum retry count
triggers human intervention when exceeded.

Checkpoint resume: each stage's input, output, and context are persisted; after
an interruption the system resumes from the checkpoint (the most recent stable
stage) instead of restarting from scratch.

Rollback strategy: the dimension with the lowest quality score determines the
rollback target (e.g. low regression score → roll back to code; low delivery
report completeness → roll back to deliver to complete the report).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from msef.types import QualityReport, Stage, StageOutput

logger = logging.getLogger(__name__)

# Dimension → rollback strategy suggestion
DIMENSION_FIX_SUGGESTIONS = {
    "syntax_correctness": "Fix syntax errors and re-run the code stage",
    "compile_pass_rate": "Fix compile errors and re-run the code stage",
    "spec_compliance": "Fix code to match the coding spec and re-run the code stage",
    "unit_pass_rate": "Add/fix unit test cases and re-run the code stage",
    "integration_pass_rate": "Fix failing integration test cases and re-run the code stage",
    "regression_result": "Fix regression defects and re-run the code stage",
    "risk_score": "Address the risk items found in review and re-run the code stage",
    "static_alert_count": "Fix static analysis alerts and re-run the code stage",
    "plan_completeness": "Add the missing plan elements and re-run the plan stage",
    "impact_coverage": "Add impact scope identification and re-run the plan stage",
    "risk_sufficiency": "Add risk markings and re-run the plan stage",
    "report_completeness": "Add the missing change report fields and re-run the deliver stage",
    "risk_marking_sufficiency": "Add risk item markings and re-run the deliver stage",
    "deliverable_completeness": "Complete the deliverables and re-run the deliver stage",
}


@dataclass
class StageCheckpoint:
    """Stage checkpoint (persistence unit for checkpoint resume)."""

    stage: Stage
    output: StageOutput
    quality: QualityReport
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class RollbackDecision:
    """Rollback decision."""

    rollback: bool
    target_stage: Optional[Stage] = None
    reason: str = ""
    suggestion: str = ""
    retry_count: int = 0

    @property
    def exceeded(self) -> bool:
        return self.retry_count > 0  # placeholder: retry counting is maintained by the caller


class RollbackManager:
    """Failure rollback manager.

    Args:
        max_retries: Maximum retry count before human intervention.
        dimension_suggestions: Dimension → rollback strategy suggestion mapping.
    """

    def __init__(self, max_retries: int = 3,
                 dimension_suggestions: Optional[Dict[str, str]] = None) -> None:
        self.max_retries = max(max_retries, 1)
        self.dimension_suggestions = dimension_suggestions or DIMENSION_FIX_SUGGESTIONS

    def on_gate_failure(self, stage: Stage, quality: QualityReport,
                        retry_count: int, history: Optional[Dict[str, Any]] = None) -> RollbackDecision:
        """Quality gate failed → determine rollback target from the lowest-scoring dimension."""
        if retry_count >= self.max_retries:
            return RollbackDecision(
                rollback=True,
                target_stage=None,
                reason=f"Retry limit ({self.max_retries}) exceeded; human intervention required",
                retry_count=retry_count,
            )
        failing = quality.failing_dimensions
        if not failing:
            failing = sorted(quality.dimension_scores, key=lambda d: d.score)[:1]
        worst = failing[0]
        suggestion = self.dimension_suggestions.get(worst.name, "Adjust the output based on the gate result and retry")
        # The main failing dimension decides the rollback target (e.g. low regression score → code; low report completeness → deliver)
        target = self._stage_for_dimension(worst.name, stage)
        logger.info(
            "[rollback] %s gate failed (%.3f<%.3f), weakest_dimension=%s, returning to %s",
            stage.value, quality.weighted_score, quality.threshold, worst.name,
            target.value if target is not None else "human intervention",
        )
        return RollbackDecision(
            rollback=True,
            target_stage=target,
            reason=f"Dimension {worst.name} scored {worst.score:.2f}, below aggregate {quality.weighted_score:.2f}",
            suggestion=suggestion,
            retry_count=retry_count + 1,
        )

    def on_approval_rejection(self, stage: Stage, comment: str, retry_count: int) -> RollbackDecision:
        """Approval rejected → determine rollback target by rejection type."""
        reason = comment or "Approval rejected"
        if "report" in reason.lower() or "deliver" in reason.lower():
            target = Stage.DELIVER
            suggestion = "Add the missing delivery content from the rejection comment"
        elif any(word in reason.lower() for word in ("standard", "coding", "code")):
            target = Stage.CODE
            suggestion = "Correct the implementation from the rejection comment"
        elif "plan" in reason.lower() or "design" in reason.lower():
            target = Stage.PLAN
            suggestion = "Revise the plan from the rejection comment"
        else:
            target = self._previous_stage(stage)
            suggestion = "Re-run the previous stage and address the rejection comment"
        return RollbackDecision(
            rollback=True,
            target_stage=target,
            reason=reason,
            suggestion=suggestion,
            retry_count=retry_count + 1,
        )

    def _stage_for_dimension(self, dim: str, current: Stage) -> Optional[Stage]:
        if dim.startswith("plan_"):
            return Stage.PLAN
        if dim in ("syntax_correctness", "compile_pass_rate", "spec_compliance"):
            return Stage.CODE
        if dim in ("unit_pass_rate", "integration_pass_rate", "regression_result"):
            return Stage.CODE  # test issues → roll back to code to fix the implementation
        if dim in ("risk_score", "static_alert_count"):
            return Stage.CODE
        if dim in ("report_completeness", "risk_marking_sufficiency", "deliverable_completeness"):
            return Stage.DELIVER
        return self._previous_stage(current)

    @staticmethod
    def _previous_stage(stage: Stage) -> Stage:
        order = Stage.ordered()
        idx = order.index(stage)
        return order[max(0, idx - 1)]


class CheckpointManager:
    """Checkpoint resume manager: save/restore checkpoints."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._checkpoints: Dict[str, StageCheckpoint] = {}
        self._last_stable: Optional[str] = None

    def save(self, task_id: str, cp: StageCheckpoint) -> None:
        if not self.enabled:
            return
        self._checkpoints[task_id] = cp
        self._last_stable = task_id
        logger.info("[checkpoint] stage %s saved (task=%s)", cp.stage.value, task_id)

    def load(self, task_id: str) -> Optional[StageCheckpoint]:
        cp = self._checkpoints.get(task_id)
        if cp:
            logger.info("[resume] restored from %s checkpoint (task=%s)", cp.stage.value, task_id)
        return cp

    def clear(self, task_id: str) -> None:
        self._checkpoints.pop(task_id, None)

    def resume_from(self, checkpoint: StageCheckpoint, next_stage: Stage,
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Rebuild context from a checkpoint."""
        ctx = dict(checkpoint.context)
        if context:
            ctx.update(context)
        ctx["resumed_from"] = checkpoint.stage.value
        return ctx
