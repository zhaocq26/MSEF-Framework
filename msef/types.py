"""Core data types for MSEF.

Python implementations of the formal concepts: development task set D,
execution stage set S, three-level memory system M, model adaptation
function φ, and execution control rules R.
"""
from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Stage(enum.Enum):
    """Execution stage: S = {plan, code, test, review, deliver}."""

    PLAN = "plan"
    CODE = "code"
    TEST = "test"
    REVIEW = "review"
    DELIVER = "deliver"

    @property
    def label(self) -> str:
        from msef.config import STAGE_LABELS

        return STAGE_LABELS[self.value]

    @property
    def index(self) -> int:
        from msef.config import STAGE_ORDER

        return STAGE_ORDER.index(self.value)

    @classmethod
    def ordered(cls) -> List["Stage"]:
        return [cls(s) for s in __import__("msef.config", fromlist=["STAGE_ORDER"]).STAGE_ORDER]


class MemoryScope(enum.Enum):
    """Memory scope: M = {project, team, personal}."""

    PROJECT = "project"
    TEAM = "team"
    PERSONAL = "personal"

    @property
    def label(self) -> str:
        return {
            "project": "project",
            "team": "team",
            "personal": "personal",
        }[self.value]


class ApprovalMode(enum.Enum):
    """Type of manual approval."""

    STAGE = "stage"      # Stage-level approval: acts on stage outputs, triggered between stages
    OPERATION = "op"     # Operation-level approval: acts on high-risk operations within a stage


class ApprovalVerdict(enum.Enum):
    """Approval verdict: approved / rejected / approved with changes."""

    APPROVED = "approved"
    REJECTED = "rejected"
    APPROVED_WITH_CHANGES = "approved_with_changes"


class OperationRiskLevel(enum.Enum):
    """Operation risk level."""

    EXTREME = "extreme"   # Extreme risk
    HIGH = "high"         # High risk
    MEDIUM = "medium"     # Medium risk
    LOW = "low"           # Low risk
    MINIMAL = "minimal"   # Minimal risk


class TaskType(enum.Enum):
    """Task category (EnterpriseDev-Bench)."""

    FEATURE = "feature"       # Feature development
    BUGFIX = "bugfix"         # Bug fix
    REFACTOR = "refactor"     # Refactoring and migration


# ---------------------------------------------------------------------------
# Tasks and subtasks
# ---------------------------------------------------------------------------
@dataclass
class Subtask:
    """The i-th subtask after decomposing task T."""

    id: str
    description: str
    files: List[str] = field(default_factory=list)
    granularity: str = "file"          # decomposition granularity (file/module_file/function_file/adaptive)
    depends_on: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)

    @classmethod
    def create(cls, description: str, files: Optional[List[str]] = None,
               granularity: str = "file", **kw: Any) -> "Subtask":
        return cls(
            id=uuid.uuid4().hex[:8],
            description=description,
            files=list(files or []),
            granularity=granularity,
            **kw,
        )


@dataclass
class Task:
    """A single complex development task T ∈ D."""

    id: str
    title: str
    description: str
    task_type: TaskType = TaskType.FEATURE
    repo_path: str = "."
    # Files actually affected by the task (extracted from VCS commit history)
    affected_files: List[str] = field(default_factory=list)
    # Risk items that must be flagged for this task
    required_risks: List[str] = field(default_factory=list)
    # Essential plan elements the task must cover (used by the planning gate)
    required_plan_elements: List[str] = field(default_factory=list)
    # Acceptance criteria (verification test cases matching the canonical fix)
    acceptance_criteria: List[str] = field(default_factory=list)
    # Extra metadata (start time, difficulty level, project name, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(cls, title: str, description: str, **kw: Any) -> "Task":
        return cls(id=uuid.uuid4().hex[:12], title=title, description=description, **kw)


@dataclass
class DecompositionResult:
    """Task decomposition result φ(T; θ_d) → subtask set."""

    subtasks: List[Subtask]
    granularity: str
    # Snapshot of decomposition strategy parameters θ_d (torch tensor, used for experience-loss backprop)
    theta: Any = None

    @property
    def count(self) -> int:
        return len(self.subtasks)


# ---------------------------------------------------------------------------
# Stage outputs and quality gates
# ---------------------------------------------------------------------------
@dataclass
class StageOutput:
    """Structured record of the k-th stage output y_k."""

    stage: Stage
    content: Dict[str, Any] = field(default_factory=dict)   # stage output content
    files: List[str] = field(default_factory=list)          # changed files involved
    meta: Dict[str, Any] = field(default_factory=dict)      # metadata (execution time, modified files, test results, etc.)
    model: Optional[str] = None                             # model that executed this stage
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "content": self.content,
            "files": self.files,
            "meta": self.meta,
            "model": self.model,
        }


@dataclass
class DimensionScore:
    """Score of a single evaluation dimension."""

    name: str
    score: float                       # normalized to [0,1], 0 worst 1 best
    detail: Dict[str, Any] = field(default_factory=dict)   # explanation of the computation

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "score": self.score, "detail": self.detail}


@dataclass
class QualityReport:
    """Quality gate result."""

    stage: Stage
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    weighted_score: float = 0.0
    threshold: float = 0.0
    passed: bool = False

    def score_map(self) -> Dict[str, float]:
        return {d.name: d.score for d in self.dimension_scores}

    @property
    def failing_dimensions(self) -> List[DimensionScore]:
        """Dimensions scoring below the weighted average (ascending), used to decide the failure rollback strategy."""
        avg = self.weighted_score
        return sorted(
            [d for d in self.dimension_scores if d.score < avg],
            key=lambda d: d.score,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "scores": [d.to_dict() for d in self.dimension_scores],
            "weighted_score": self.weighted_score,
            "threshold": self.threshold,
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# Manual approval
# ---------------------------------------------------------------------------
@dataclass
class Operation:
    """High-risk operation op_{k,j} inside a stage."""

    id: str
    stage: Stage
    name: str
    description: str = ""
    risk_level: OperationRiskLevel = OperationRiskLevel.MEDIUM
    # Impact scope of the operation (number of modules involved + number of callers)
    impact_scope: int = 1
    reversibility: float = 0.5          # 1=fully reversible 0=irreversible
    trigger: str = ""

    @classmethod
    def create(cls, stage: Stage, name: str, **kw: Any) -> "Operation":
        return cls(id=uuid.uuid4().hex[:8], stage=stage, name=name, **kw)


@dataclass
class ApprovalRequest:
    """Approval request (submission step of the approval flow)."""

    id: str
    mode: ApprovalMode
    stage: Stage
    payload: Dict[str, Any] = field(default_factory=dict)      # content awaiting approval
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    approver: str = ""                       # approving entity
    submitted_at: float = field(default_factory=time.time)
    result: Optional[ApprovalVerdict] = None
    comment: str = ""
    resolved_at: Optional[float] = None

    @classmethod
    def create(cls, mode: ApprovalMode, stage: Stage, **kw: Any) -> "ApprovalRequest":
        return cls(id=uuid.uuid4().hex[:12], mode=mode, stage=stage, **kw)


# ---------------------------------------------------------------------------
# Memory system
# ---------------------------------------------------------------------------
@dataclass
class MemoryEntry:
    """Memory entry: triple (content, scope, vector) plus metadata."""

    id: str
    content: str
    scope: MemoryScope
    # Semantic vector (PyTorch float tensor or numpy array)
    vector: Any = None
    category: str = "general"               # category (architecture design / coding standards / startup method / release requirements, ...)
    source_type: str = "manual"             # source: stage_output/approval/rollback/manual
    created_at: float = field(default_factory=time.time)
    last_access_at: float = field(default_factory=time.time)
    usage_count: int = 0
    consistency: float = 1.0
    archived: bool = False
    downgraded: bool = False
    # Version history (version control)
    versions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "scope": self.scope.value,
            "category": self.category,
            "source_type": self.source_type,
            "created_at": self.created_at,
            "last_access_at": self.last_access_at,
            "usage_count": self.usage_count,
            "consistency": self.consistency,
            "archived": self.archived,
            "downgraded": self.downgraded,
        }


# ---------------------------------------------------------------------------
# Execution records
# ---------------------------------------------------------------------------
@dataclass
class StageExecutionRecord:
    """Stage execution record (written into the memory system / used for collaborative optimization)."""

    task_id: str
    stage: Stage
    quality: QualityReport
    output: StageOutput
    retries: int = 0
    rolled_back: bool = False
    approvals: List[ApprovalRequest] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    duration_seconds: float = 0.0

    def performance(self) -> float:
        """Actual performance score of this stage (source of the y value for the experience loss function g(T, θ_d))."""
        return self.quality.weighted_score


@dataclass
class TaskExecutionResult:
    """Overall execution result of a single task."""

    task: Task
    task_type: str = "feature"
    subtasks: List[Subtask] = field(default_factory=list)
    stage_records: List[StageExecutionRecord] = field(default_factory=list)
    # Output snapshots of the five stages: plan/code/test/review/deliver
    stage_outputs: Dict[str, StageOutput] = field(default_factory=dict)
    quality_reports: Dict[str, QualityReport] = field(default_factory=dict)
    total_retries: int = 0
    total_approvals: int = 0
    human_interventions: int = 0
    completed: bool = False
    final_score: float = 0.0
    # Task-level exact-match result, separate from stage gate quality.
    exact_match: float = 0.0
    rollback_stages: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    token_usage: int = 0
    # Checkpoint resume
    last_checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.id,
            "task_type": self.task_type,
            "completed": self.completed,
            "final_score": self.final_score,
            "exact_match": self.exact_match,
            "rollback_stages": list(self.rollback_stages),
            "duration_seconds": self.duration_seconds,
            "total_retries": self.total_retries,
            "total_approvals": self.total_approvals,
            "human_interventions": self.human_interventions,
            "token_usage": self.token_usage,
        }
