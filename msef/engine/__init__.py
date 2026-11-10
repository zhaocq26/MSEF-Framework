"""Multi-stage execution engine.

Splits a complex development task into five traceable stages
(plan, code, test, review, deliver) with quality gates, manual
approval, failure rollback / checkpoint resume, and dependency
orchestration for structured execution control.
"""
from msef.engine.approval import (
    ApprovalResult,
    ApprovalService,
    HumanApprover,
    OperationRiskEvaluator,
    StageRiskEvaluator,
)
from msef.engine.engine import MultiStageExecutionEngine
from msef.engine.orchestration import (
    DependencyOrchestrator,
    StageDependencyGraph,
)
from msef.engine.quality_gate import QualityGate, QualityGateService
from msef.engine.rollback import CheckpointManager, RollbackManager

__all__ = [
    "ApprovalResult",
    "ApprovalService",
    "HumanApprover",
    "OperationRiskEvaluator",
    "StageRiskEvaluator",
    "MultiStageExecutionEngine",
    "DependencyOrchestrator",
    "StageDependencyGraph",
    "QualityGate",
    "QualityGateService",
    "CheckpointManager",
    "RollbackManager",
]