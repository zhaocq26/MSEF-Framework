"""Multi-stage execution engine.

Closed-loop architecture: the engine controls staged execution, the memory
system accumulates knowledge, collaborative optimization provides feedback,
and the model adaptation layer supplies underlying capability. The engine
advances tasks stage by stage, running quality gates and human approvals at
each stage boundary, and triggers failure rollback with checkpoint resume on
failure.

Formalized execution:

    Task T → adaptation-layer reduction φ(T) → engine advances by stage:
        y_k = e_k(φ(T), M)
        Q(y_k) ≥ τ_k  → gate passed, move to next stage (memory update M ← M ∪ y_k)
        Q(y_k) <  τ_k  → failure rollback to the last stable stage and re-execute
    Key decision points trigger human approval (stage-level / operation-level)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from msef.adaptation.compression import ContextCompressor
from msef.adaptation.decomposition import TaskReducer
from msef.adaptation.router import ModelRouter
from msef.agent.agent import Agent
from msef.config import Config
from msef.engine.approval import ApprovalService, ApprovalVerdict
from msef.engine.operations import OperationDetector
from msef.engine.orchestration import DependencyOrchestrator
from msef.engine.quality_gate import QualityGateService
from msef.engine.rollback import CheckpointManager, RollbackManager, StageCheckpoint
from msef.memory.entry import MemoryEntry
from msef.memory.retrieval import MemoryRetriever
from msef.memory.storage import MemoryStore
from msef.types import (
    ApprovalRequest,
    MemoryScope,
    Stage,
    StageExecutionRecord,
    StageOutput,
    Task,
    TaskExecutionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineMetrics:
    """Engine runtime metrics (SPR/RBR/HIR/TSR/EM/ACT/Token).

    stage_attempts: total stage attempts
    stage_passed_first: stages passing the gate on the first attempt (SPR numerator)
    """
    stage_attempts: int = 0
    stage_passed_first: int = 0          # stages passing the gate on the first attempt (SPR numerator)
    rollbacks: int = 0
    approvals: int = 0
    human_interventions: int = 0
    token_usage: int = 0
    completed_tasks: int = 0
    total_duration: float = 0.0
    exact_match_total: float = 0.0

    @property
    def stage_pass_rate(self) -> float:
        """Stage pass rate (SPR)."""
        return self.stage_passed_first / self.stage_attempts if self.stage_attempts else 0.0

    @property
    def rollback_rate(self) -> float:
        """Rollback rate (RBR)."""
        return self.rollbacks / max(1, self.completed_tasks + self.rollbacks)

    @property
    def avg_duration(self) -> float:
        return self.total_duration / max(1, self.completed_tasks)

    @property
    def exact_match_rate(self) -> float:
        return self.exact_match_total / max(1, self.completed_tasks)


class MultiStageExecutionEngine:
    """MSEF multi-stage execution engine.

    Args:
        agent: The agent (with LLM backend).
        reducer: Task reducer (model adaptation layer).
        compressor: Context pre-compressor.
        router: Multi-model router.
        gates: Quality gate service.
        approvals: Human approval service.
        rollback: Failure rollback manager.
        checkpoint: Checkpoint resume manager.
        store: Three-tier memory store.
        retriever: Memory retriever.
        memory_write: Optional memory write hook callable(task_id, stage, output, report) ->
            List[MemoryEntry] (execution-engine → memory-system feedback).
        config: Global config.
        orchestrator: Dependency orchestrator.
    """

    def __init__(
        self,
        agent: Agent,
        reducer: TaskReducer,
        compressor: ContextCompressor,
        router: ModelRouter,
        gates: QualityGateService,
        approvals: ApprovalService,
        rollback: RollbackManager,
        checkpoint: CheckpointManager,
        store: MemoryStore,
        retriever: MemoryRetriever,
        memory_write: Optional[Callable[..., List[MemoryEntry]]] = None,
        config: Optional[Config] = None,
        orchestrator: Optional[DependencyOrchestrator] = None,
    ) -> None:
        self.agent = agent
        self.reducer = reducer
        self.compressor = compressor
        self.router = router
        self.gates = gates
        self.approvals = approvals
        self.rollback = rollback
        self.checkpoint = checkpoint
        self.store = store
        self.retriever = retriever
        self.memory_write = memory_write or self._default_memory_write
        self.config = config or Config()
        self.orchestrator = orchestrator or DependencyOrchestrator()
        self.op_detector = OperationDetector(project_total_scope=100)
        self.metrics = EngineMetrics()
        # Approval records → memory-system feedback (persist human approval records)
        if self.approvals.config.record_to_memory and self.approvals.on_approval_record is None:
            self.approvals.on_approval_record = self._default_approval_memory_write

    # ------------------------------------------------------------------
    def _default_approval_memory_write(self, request: ApprovalRequest) -> List[MemoryEntry]:
        """Persist approval requests/results/risk assessments to project-level memory."""
        doc = (
            f"[Approval record] mode={request.mode.value} stage={request.stage.value}\n"
            f"approver={request.approver} result={request.result.value if request.result else 'pending'}\n"
            f"comment={request.comment}\n"
            f"risk_assessment={request.risk_assessment}"
        )
        entry = self.store.add(
            doc, MemoryScope.PROJECT,
            category="approval", source_type="approval",
        )
        return [entry]

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self, task: Task, contexts: Optional[List[str]] = None,
            resume: bool = True) -> TaskExecutionResult:
        """Execute a complex development task and return the full result."""
        result = TaskExecutionResult(task=task, task_type=task.task_type.value)
        logger.info("═══ Starting task %s: %s ═══", task.id, task.title)
        # Snapshot shared approval counters; diff at task end for this task's counts
        _approvals_before = len(self.approvals.requests)
        _interventions_before = self.approvals.human_interventions

        # 1. Model adaptation layer: task reduction + context pre-compression + memory mount
        decomposition = self.reducer.decompose(task)
        result.subtasks = decomposition.subtasks
        compressed = self.compressor.compress(
            contexts or [], task,
            included_files=task.affected_files,
        )

        # 2. Memory retrieval and mount
        query = f"{task.title} {task.description}"
        mounted, entries = self.retriever.mount(query)

        # 3. Checkpoint resume detection
        cp = self.checkpoint.load(task.id) if resume else None
        schedule = self.orchestrator.schedule()
        start_idx = cp.stage.index + 1 if cp else 0
        prior_outputs: Dict[Stage, StageOutput] = {}
        if cp is not None:
            prior_outputs[cp.stage] = cp.output
            result.stage_outputs[cp.stage.value] = cp.output

        # 4. Five-stage execution: stage advance + quality gates + human approval + failure rollback
        retries_by_stage: Dict[str, int] = {}
        rollback_count = 0
        # Infinite-loop guard: total rollback budget = max retries × number of stages
        rollback_budget = self.config.max_retries * len(schedule) + 2
        feedback = ""  # adjustment suggestions from rollbacks/rejections, injected into later context

        i = start_idx
        while i < len(schedule):
            stage = schedule[i]
            # On rollback, discard current and subsequent stage outputs
            for s in schedule[i:]:
                result.stage_outputs.pop(s.value, None)
                result.quality_reports.pop(s.value, None)
                result.stage_records = [r for r in result.stage_records if r.stage != s]

            stage_key = stage.value
            if retries_by_stage.get(stage_key, 0) >= self.config.max_retries:
                logger.error("[task] %s %s stage exceeded retry limit; escalating to a human",
                             task.id, stage.value)
                result.last_checkpoint = "human_escalation"
                break

            self.metrics.stage_attempts += 1
            attempt = retries_by_stage.get(stage_key, 0)
            model = self.router.route(stage, task)
            output = self.agent.execute(
                stage, task,
                subtasks=decomposition.subtasks,
                compressed_context=compressed + feedback,
                mounted_memory=mounted,
                prior_outputs=prior_outputs,
                model=model,
            )
            task_context = self._task_context(task)
            report = self.gates.evaluate(stage, output, task_context)

            # Operation-level approval (high-risk operations inside the stage)
            op_requests = self._run_operation_approvals(output)
            # Stage-level approval (review/deliver always trigger)
            stage_request = self._run_stage_approval(stage, output, report, task)
            approvals = op_requests + ([stage_request] if stage_request else [])
            self.metrics.approvals += len(approvals)

            if not report.passed:
                # Quality gate failed → failure rollback
                result.rollback_stages.append(stage.value)
                decision = self.rollback.on_gate_failure(stage, report, attempt)
                self.metrics.rollbacks += 1
                rollback_count += 1
                retries_by_stage[stage_key] = attempt + 1
                self._record(result, task, stage, output, report, attempt, approvals,
                             rolled_back=True)
                if decision.target_stage is None or rollback_count > rollback_budget:
                    logger.error(
                        "[task] %s %s gate failed after retry limit; task terminated", task.id, stage.value,
                    )
                    result.last_checkpoint = "gate_failure_escalation"
                    break
                logger.info(
                    "[rollback] %s %s gate failed -> returning to %s (attempt %d)",
                    task.id, stage.label, decision.target_stage.label, attempt + 1,
                )
                feedback = f"\n(Rollback suggestion: {decision.suggestion})"
                i = schedule.index(decision.target_stage)
                continue

            # Gate passed but stage-level approval rejected → rollback
            if stage_request is not None and stage_request.result == ApprovalVerdict.REJECTED:
                decision = self.rollback.on_approval_rejection(stage, stage_request.comment, attempt)
                self.metrics.rollbacks += 1
                rollback_count += 1
                retries_by_stage[stage_key] = attempt + 1
                self._record(result, task, stage, output, report, attempt, approvals,
                             rolled_back=True)
                if decision.target_stage is None or rollback_count > rollback_budget:
                    result.last_checkpoint = "approval_rejection_escalation"
                    break
                logger.info(
                    "[rollback] %s %s approval rejected -> returning to %s",
                    task.id, stage.label, decision.target_stage.label,
                )
                feedback = f"\n(Rejection comment: {stage_request.comment}; adjustment: {decision.suggestion})"
                i = schedule.index(decision.target_stage)
                continue

            # Passed: record, write memory, save checkpoint, advance to next stage
            self.metrics.stage_passed_first += 1 if attempt == 0 else 0
            self._record(result, task, stage, output, report, attempt, approvals)
            prior_outputs[stage] = output
            record = result.stage_records[-1]
            self.memory_write(task, stage, output, record.quality)
            self.checkpoint.save(task.id, StageCheckpoint(
                stage=stage, output=output, quality=record.quality,
                context={"prior": {s.value: prev.to_dict()
                                   for s, prev in prior_outputs.items()}},
                created_at=time.time(),
            ))
            i += 1
        else:
            result.completed = True

        result.end_time = time.time()
        result.duration_seconds = result.end_time - result.start_time
        result.token_usage = self.agent.token_usage
        result.human_interventions = self.approvals.human_interventions - _interventions_before
        result.total_approvals = len(self.approvals.requests) - _approvals_before
        result.total_retries = rollback_count

        passed_quals = [r.quality for r in result.stage_records]
        result.final_score = (
            sum(q.weighted_score for q in passed_quals) / len(passed_quals)
            if passed_quals else 0.0
        )

        # Exact match is a task-level benchmark signal supplied by the
        # backend's delivery report, distinct from the stage gate score.
        deliver_output = result.stage_outputs.get(Stage.DELIVER.value)
        if deliver_output is not None:
            result.exact_match = float(deliver_output.content.get("exact_match", 0.0))

        if result.completed:
            self.metrics.completed_tasks += 1
            self.metrics.total_duration += result.duration_seconds
            self.metrics.exact_match_total += result.exact_match
        logger.info("═══ Task %s completed=%s final_score=%.3f duration=%.1fs rollbacks=%d ═══",
                    task.id, result.completed, result.final_score,
                    result.duration_seconds, rollback_count)
        return result

    # ------------------------------------------------------------------
    def _run_operation_approvals(self, output: StageOutput) -> List[ApprovalRequest]:
        """Detect high-risk operations inside the stage output and run operation-level approval."""
        requests: List[ApprovalRequest] = []
        for op in self.op_detector.detect(output):
            request = self.approvals.check_operation_approval(op)
            if request is not None:
                # Default approve (swappable with a real approval system); rejections are set in tests
                self.approvals.decide(request)
                requests.append(request)
        return requests

    def _run_stage_approval(self, stage: Stage, output: StageOutput,
                            report, task: Task) -> Optional[ApprovalRequest]:
        """Stage-level approval (review/deliver always trigger by default)."""
        # Risk items: review stage uses risk_items, other stages use the risks list
        raw = output.content.get("risk_items") or [
            {"level": "high", "name": r} for r in (output.content.get("risks") or [])
        ]
        risk_items = [
            r if isinstance(r, dict) else {"level": "medium", "name": r}
            for r in raw
        ][:3]
        touches_critical = self._touches_critical(task, output)
        request = self.approvals.check_stage_approval(stage, output, risk_items,
                                                      touches_critical=touches_critical)
        if request is not None:
            self.approvals.decide(request)
        return request

    def _touches_critical(self, task: Task, output: StageOutput) -> bool:
        critical = [str(c) for c in task.metadata.get("critical_modules", [])]
        if not critical:
            return False
        for f in output.files:
            for c in critical:
                # Critical modules are path prefixes (e.g. "src/auth" covers "src/auth/login.py")
                if f == c or f.startswith(c.rstrip("/") + "/") or f.startswith(c):
                    return True
        return False

    # ------------------------------------------------------------------
    def _task_context(self, task: Task) -> Dict[str, Any]:
        return {
            "required_plan_elements": task.required_plan_elements or self._default_plan_elements(),
            "affected_files": task.affected_files,
            "required_risks": task.required_risks,
        }

    @staticmethod
    def _default_plan_elements() -> List[str]:
        return ["scope", "dependencies", "sequence", "risks", "rollback"]

    def _record(self, result: TaskExecutionResult, task: Task, stage: Stage,
                output: StageOutput, report, attempt: int,
                approvals: List[ApprovalRequest], rolled_back: bool = False) -> None:
        rec = StageExecutionRecord(
            task_id=task.id,
            stage=stage,
            quality=report,
            output=output,
            retries=attempt,
            rolled_back=rolled_back,
            approvals=approvals,
            duration_seconds=0.0,
        )
        result.stage_records.append(rec)
        result.stage_outputs[stage.value] = output
        result.quality_reports[stage.value] = report

    # ------------------------------------------------------------------
    # Memory write (execution engine → memory-system feedback)
    # ------------------------------------------------------------------
    def _default_memory_write(self, task: Task, stage: Stage, output: StageOutput,
                              report) -> List[MemoryEntry]:
        """Write stage output and gate results to project-level memory."""
        doc = (
            f"[Task {task.id} · {stage.label} stage output]\n"
            f"task: {task.title}\n"
            f"gate_score: {report.weighted_score:.3f} (threshold {report.threshold})\n"
            f"output_summary: {str(output.content)[:500]}"
        )
        entry = self.store.add(
            doc, MemoryScope.PROJECT,
            category=f"stage_{stage.value}",
            source_type="stage_output",
        )
        return [entry]
