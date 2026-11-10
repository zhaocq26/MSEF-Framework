"""Multi-stage execution engine end-to-end tests (five-stage flow, quality gates, human approval,
failure rollback, checkpoint resume, memory writes)."""
import pytest

from msef.adaptation.compression import ContextCompressor
from msef.adaptation.decomposition import TaskReducer
from msef.adaptation.router import ModelRouter
from msef.agent.agent import Agent
from msef.config import Config
from msef.engine.approval import ApprovalService
from msef.engine.engine import MultiStageExecutionEngine
from msef.engine.quality_gate import QualityGateService
from msef.engine.rollback import CheckpointManager, RollbackManager
from msef.llm.simulated import SimulatedBackend
from msef.memory.retrieval import MemoryRetriever
from msef.memory.storage import MemoryStore
from msef.types import ApprovalVerdict, MemoryScope, Stage, Task, TaskType


# ---------------------------------------------------------------------------
# Helpers for building the engine
# ---------------------------------------------------------------------------
def build_engine(profile, tmp_path, config=None, approvals=None, **kw):
    cfg = config or Config()
    store = MemoryStore(
        project_root=str(tmp_path / "project"),
        team_dir=str(tmp_path / "team"),
        user_home=str(tmp_path / "user"),
        persist=False,
    )
    retriever = MemoryRetriever(store, cfg.memory)
    backend = SimulatedBackend(profile=profile, seed=42)
    agent = Agent(backend=backend)
    reducer = TaskReducer(policy=None, seed=42)
    compressor = ContextCompressor(cfg.adaptation)
    router = ModelRouter(cfg.adaptation)
    gates = QualityGateService(cfg)
    approval_svc = approvals or ApprovalService()
    rollback = RollbackManager(max_retries=kw.get("max_retries", 3))
    checkpoint = CheckpointManager(enabled=kw.get("checkpoint_enabled", True))
    engine = MultiStageExecutionEngine(
        agent=agent,
        reducer=reducer,
        compressor=compressor,
        router=router,
        gates=gates,
        approvals=approval_svc,
        rollback=rollback,
        checkpoint=checkpoint,
        store=store,
        retriever=retriever,
        config=cfg,
    )
    return engine, store, agent


HAPPY_PROFILE = {
    "plan": {"plan_completeness": 1.0, "impact_coverage": 1.0, "risk_sufficiency": 1.0,
             "files": ["src/auth/login.py"], "risks": ["authentication logic change"]},
    "code": {"syntax_ok": True, "compile_ok": True, "spec_ok": 1.0,
             "files": ["src/auth/login.py"]},
    "test": {"unit_pass": 0.95, "integration_pass": 0.95, "new_defects": 0.0},
    "review": {"spec_ok": 1.0, "risk_items": [{"level": "low"}], "alerts": 1},
    "deliver": {"risks": ["authentication logic change"]},
}


def make_task(**kw) -> Task:
    meta = {"complexity": 0.5}
    if "metadata" in kw:
        meta.update(kw.pop("metadata"))
    return Task.create(
        title="Add captcha validation to the login API",
        description="Add sliding-captcha validation to the login endpoint and update request parameters.",
        task_type=TaskType.FEATURE,
        affected_files=["src/auth/login.py"],
        required_risks=["authentication logic change"],
        required_plan_elements=["scope", "dependencies", "sequence", "risks", "rollback"],
        metadata=meta,
        **kw,
    )


def test_happy_path_five_stages_complete(tmp_path):
    """All five stages pass their quality gates → task completes with full outputs."""
    engine, store, agent = build_engine(HAPPY_PROFILE, tmp_path)
    task = make_task()
    result = engine.run(task, contexts=["src/auth/login.py login endpoint implementation"])
    assert result.completed
    assert set(result.stage_outputs.keys()) >= {"plan", "code", "test", "review", "deliver"}
    assert result.total_retries == 0
    assert result.final_score > 0.7
    # Stage outputs are written to memory (engine → memory feedback)
    assert store.count(MemoryScope.PROJECT) >= 5


def test_plan_gate_failure_triggers_rollback_then_success(tmp_path):
    """Plan gate failure triggers rollback, then retry succeeds.

    The first plan output lacks required elements (empty plan_elements → low plan completeness →
    gate failure); after rollback injects adjustment suggestions, the simulated agent fills them
    in and the retry passes.
    """
    profile = {
        "plan": {"plan_completeness": 0.4, "impact_coverage": 1.0, "risk_sufficiency": 0.5,
                 "files": ["src/auth/login.py"], "risks": ["authentication logic change"],
                 "plan_elements": []},                      # first round: incomplete plan
        "code": {"syntax_ok": True, "compile_ok": True, "spec_ok": 1.0,
                 "files": ["src/auth/login.py"]},
        "test": {"unit_pass": 0.95, "integration_pass": 0.95, "new_defects": 0.0},
        "review": {"spec_ok": 1.0, "risk_items": [{"level": "low"}], "alerts": 1},
        "deliver": {"risks": ["authentication logic change"]},
    }
    retry_profile = {
        "plan": {"plan_elements": ["scope", "dependencies", "sequence", "risks", "rollback"]},
    }
    cfg = Config(max_retries=3)
    store = MemoryStore(
        project_root=str(tmp_path / "project"), team_dir=str(tmp_path / "team"),
        user_home=str(tmp_path / "user"), persist=False,
    )
    retriever = MemoryRetriever(store, cfg.memory)
    backend = SimulatedBackend(profile=profile, retry_profile=retry_profile, seed=42)
    agent = Agent(backend=backend)
    engine = MultiStageExecutionEngine(
        agent=agent, reducer=TaskReducer(seed=42),
        compressor=ContextCompressor(cfg.adaptation), router=ModelRouter(cfg.adaptation),
        gates=QualityGateService(cfg), approvals=ApprovalService(),
        rollback=RollbackManager(max_retries=3), checkpoint=CheckpointManager(),
        store=store, retriever=retriever, config=cfg,
    )
    task = make_task()
    result = engine.run(task)
    assert result.completed
    assert result.total_retries >= 1
    assert engine.metrics.rollbacks >= 1
    assert result.quality_reports["plan"].passed


def test_test_failure_rolls_back_to_code(tmp_path):
    """Test-stage regression failure rolls back to the code stage to fix the implementation."""
    profile = {
        "plan": {"plan_completeness": 1.0, "impact_coverage": 1.0, "risk_sufficiency": 1.0,
                 "files": ["src/auth/login.py"], "risks": ["authentication logic change"]},
        "code": {"syntax_ok": True, "compile_ok": True, "spec_ok": 1.0,
                 "files": ["src/auth/login.py"]},
        "test": {"unit_pass": 0.5, "integration_pass": 0.5, "new_defects": 0.5},  # low regression
        "review": {"spec_ok": 1.0, "risk_items": [{"level": "low"}], "alerts": 1},
        "deliver": {"risks": ["authentication logic change"]},
    }
    retry_profile = {
        "test": {"unit_pass": 0.95, "integration_pass": 0.95, "new_defects": 0.0},
    }
    cfg = Config(max_retries=3)
    store = MemoryStore(
        project_root=str(tmp_path / "project"), team_dir=str(tmp_path / "team"),
        user_home=str(tmp_path / "user"), persist=False,
    )
    retriever = MemoryRetriever(store, cfg.memory)
    backend = SimulatedBackend(profile=profile, retry_profile=retry_profile, seed=42)
    agent = Agent(backend=backend)
    engine = MultiStageExecutionEngine(
        agent=agent, reducer=TaskReducer(seed=42),
        compressor=ContextCompressor(cfg.adaptation), router=ModelRouter(cfg.adaptation),
        gates=QualityGateService(cfg), approvals=ApprovalService(),
        rollback=RollbackManager(max_retries=3), checkpoint=CheckpointManager(),
        store=store, retriever=retriever, config=cfg,
    )
    task = make_task()
    result = engine.run(task)
    assert result.completed
    assert result.total_retries >= 1
    # After rollback, code and test stages re-run → final test gate passes
    assert result.quality_reports["test"].passed
    assert result.quality_reports["code"].passed


def test_retry_exhaustion_marks_task_incomplete(tmp_path):
    """Persistent code-stage failure beyond max retries leaves the task incomplete."""
    profile = {
        "plan": {"plan_completeness": 1.0, "impact_coverage": 1.0, "risk_sufficiency": 1.0,
                 "files": ["src/auth/login.py"], "risks": ["authentication logic change"]},
        "code": {"syntax_ok": False, "compile_ok": False, "spec_ok": 0.5,  # syntax keeps failing
                 "files": ["src/auth/login.py"]},
        "test": {"unit_pass": 0.95, "integration_pass": 0.95, "new_defects": 0.0},
        "review": {"spec_ok": 1.0, "risk_items": [{"level": "low"}], "alerts": 1},
        "deliver": {"risks": ["authentication logic change"]},
    }
    engine, store, _ = build_engine(profile, tmp_path, max_retries=2)
    result = engine.run(make_task())
    assert not result.completed
    assert result.total_retries >= 1
    assert engine.metrics.stage_pass_rate < 1.0


def test_stage_approval_triggers_and_records(tmp_path):
    """Review touching critical modules triggers stage-level approval and records it to memory."""
    records = []

    class RecordingApprover:
        def approve(self, request):
            records.append(request)
            from msef.engine.approval import ApprovalResult

            return ApprovalResult(request.id, ApprovalVerdict.APPROVED, "approved; standards satisfied")

    svc = ApprovalService(
        approver=RecordingApprover(),
        critical_modules=["src/auth"],
        project_total_files=1,
    )
    engine, store, _ = build_engine(HAPPY_PROFILE, tmp_path, approvals=svc)
    # Task touches critical module src/auth
    task = make_task(metadata={"complexity": 0.5, "critical_modules": ["src/auth"]})
    result = engine.run(task)
    assert result.completed
    assert len(records) >= 1
    assert svc.human_interventions >= 1
    # Approval records persisted to memory (human approval record source)
    approval_entries = [e for e in store.entries() if e.source_type == "approval"]
    assert approval_entries


def test_operation_approval_detected(tmp_path):
    """High-risk operations in plan output trigger operation-level approval before execution."""
    profile = {
        "plan": {"plan_completeness": 1.0, "impact_coverage": 1.0, "risk_sufficiency": 1.0,
                 "files": ["src/auth/login.py"], "risks": ["authentication logic change"],
                 "operations": [
                     {"name": "modify database schema", "risk_level": "extreme",
                      "impact_scope": 8, "reversibility": 0.1},
                 ]},
        "code": {"syntax_ok": True, "compile_ok": True, "spec_ok": 1.0,
                 "files": ["src/auth/login.py"]},
        "test": {"unit_pass": 0.95, "integration_pass": 0.95, "new_defects": 0.0},
        "review": {"spec_ok": 1.0, "risk_items": [{"level": "low"}], "alerts": 1},
        "deliver": {"risks": ["authentication logic change"]},
    }
    svc = ApprovalService()
    engine, store, _ = build_engine(profile, tmp_path, approvals=svc)
    result = engine.run(make_task())
    assert result.completed
    op_approvals = [r for r in svc.requests if r.mode.value == "op"]
    assert len(op_approvals) >= 1


def test_checkpoint_resume(tmp_path):
    """Checkpoint resume: after saving a stage checkpoint, execution resumes from the interruption point."""
    engine, store, agent = build_engine(HAPPY_PROFILE, tmp_path)
    task = make_task()
    result = engine.run(task)
    assert result.completed
    assert engine.checkpoint.load(task.id) is not None

    # Simulate interrupted resume: run a new task halfway, then continue from a checkpoint
    engine2, store2, _ = build_engine(HAPPY_PROFILE, tmp_path)
    task2 = make_task()
    # Preset a "code stage" checkpoint (simulating plan + code already done before interruption)
    from msef.engine.rollback import StageCheckpoint
    from msef.types import StageOutput

    cp = StageCheckpoint(
        stage=Stage.CODE,
        output=StageOutput(stage=Stage.CODE, content={"syntax_ok": True}, files=["src/auth/login.py"]),
        quality=result.quality_reports["code"],
    )
    engine2.checkpoint.save(task2.id, cp)
    result2 = engine2.run(task2)
    assert result2.completed
    # Continue from code onward: test/review/deliver records exist
    assert "test" in result2.quality_reports
    assert "deliver" in result2.quality_reports


def test_memory_boosted_retrieval_context(tmp_path):
    """Mounted memory content enters the agent prompt (context mounting)."""
    engine, store, _ = build_engine(HAPPY_PROFILE, tmp_path)
    # Preseed project-level memory
    store.add("The login module uses the sliding captcha component captcha-sdk.", MemoryScope.PROJECT, category="architecture")
    task = make_task()
    result = engine.run(task)
    assert result.completed
    # Entry is accessed during retrieval (usage_count increases)
    entries = [e for e in store.entries() if e.category == "architecture"]
    assert entries and entries[0].last_access_at > 0


def test_stage_outputs_structure(tmp_path):
    """Stage outputs are structured records (y_k as a structured record)."""
    engine, store, _ = build_engine(HAPPY_PROFILE, tmp_path)
    result = engine.run(make_task())
    code_out = result.stage_outputs["code"]
    assert code_out.stage == Stage.CODE
    assert code_out.model is not None
    assert "syntax_ok" in code_out.content
    assert result.token_usage > 0
