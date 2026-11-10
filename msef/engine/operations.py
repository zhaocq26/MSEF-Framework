"""High-risk operation detection.

Detects high-risk operations from the ``operations`` field of stage outputs
(plan/code), converts them to ``Operation`` objects, and hands them to
operation-level approval for pre-execution review. Risk levels align with the
operation risk level table.
"""
from __future__ import annotations

from typing import Any, Dict, List

from msef.types import Operation, OperationRiskLevel, Stage

LEVEL_MAP = {
    "extreme": OperationRiskLevel.EXTREME,
    "high": OperationRiskLevel.HIGH,
    "medium": OperationRiskLevel.MEDIUM,
    "low": OperationRiskLevel.LOW,
    "minimal": OperationRiskLevel.MINIMAL,
}


class OperationDetector:
    """Extract high-risk operations from stage outputs."""

    def __init__(self, project_total_scope: int = 100) -> None:
        self.project_total_scope = max(1, project_total_scope)

    def detect(self, output: StageOutput) -> List[Operation]:
        ops: List[Operation] = []
        raw_ops = output.content.get("operations") or []
        for i, op in enumerate(raw_ops):
            if isinstance(op, str):
                ops.append(self._from_name(f"op_{i}", output.stage, op))
            elif isinstance(op, dict):
                ops.append(self._from_dict(f"op_{i}", output.stage, op))
        # Fallback: planned risks mentioning schema/config keywords also count as high-risk operations.
        for risk in output.content.get("risks") or []:
            r = str(risk)
            if any(kw in r.lower() for kw in ("config", "schema", "database", "api", "dependency", "delete", "merge", "deploy")):
                ops.append(self._from_name(f"risk_op_{len(ops)}", output.stage, r, risk_level="high", trigger=r))
        return ops

    def _from_dict(self, op_id: str, stage: Stage, d: Dict[str, Any]) -> Operation:
        name = d.get("name", op_id)
        level = LEVEL_MAP.get(str(d.get("risk_level", "medium")).lower(), OperationRiskLevel.MEDIUM)
        return Operation(
            id=op_id,
            stage=stage,
            name=name,
            description=d.get("description", ""),
            risk_level=level,
            impact_scope=int(d.get("impact_scope", 1)),
            reversibility=float(d.get("reversibility", 0.5)),
            trigger=d.get("trigger", ""),
        )

    def _from_name(self, op_id: str, stage: Stage, name: str,
                   risk_level: str = "medium", trigger: str = "") -> Operation:
        return Operation(
            id=op_id,
            stage=stage,
            name=name,
            risk_level=LEVEL_MAP.get(risk_level.lower(), OperationRiskLevel.MEDIUM),
            impact_scope=1,
            reversibility=0.5,
            trigger=trigger or name,
        )
