"""Human approval tests (stage-level approval, operation-level approval, risk functions, approval flow)."""
from msef.config import ApprovalConfig
from msef.engine.approval import (
    ApprovalService,
    HumanApprover,
    OperationRiskEvaluator,
    StageRiskEvaluator,
)
from msef.types import (
    ApprovalMode,
    ApprovalVerdict,
    Operation,
    OperationRiskLevel,
    Stage,
    StageOutput,
)


def make_review_output() -> StageOutput:
    return StageOutput(stage=Stage.REVIEW, files=["auth.py", "core.py"],
                       content={"risks": [{"level": "high"}, {"level": "low"}]})


def test_stage_risk_scale_grows_with_files():
    """Stage-level risk: more changed files / touching critical modules / higher high-risk share → higher risk."""
    evaluator = StageRiskEvaluator(project_total_files=100)
    small = StageOutput(stage=Stage.REVIEW, files=["a.py"])
    big = StageOutput(stage=Stage.REVIEW, files=[f"f{i}.py" for i in range(50)])
    r_small = evaluator.evaluate(small, risk_items=[])
    r_big = evaluator.evaluate(big, risk_items=[{"level": "high"}])
    assert r_big > r_small

    touch = evaluator.evaluate(big, risk_items=[], touches_critical=True)
    assert touch >= r_big

    # High-risk share
    many_high = evaluator.evaluate(small, risk_items=[{"level": "high"}, {"level": "high"}])
    few_high = evaluator.evaluate(small, risk_items=[{"level": "high"}, {"level": "low"}])
    assert many_high >= few_high


def test_operation_risk_levels_table5():
    """Default weights for extreme/high/medium/low risk levels decrease monotonically."""
    evaluator = OperationRiskEvaluator(project_total_scope=100)
    op_extreme = Operation.create(Stage.CODE, "modify production configuration", risk_level=OperationRiskLevel.EXTREME,
                                  impact_scope=2, reversibility=0.0)
    op_minimal = Operation.create(Stage.CODE, "read file", risk_level=OperationRiskLevel.MINIMAL,
                                  impact_scope=1, reversibility=1.0)
    assert evaluator.evaluate(op_extreme) > evaluator.evaluate(op_minimal)


def test_operation_approval_trigger_threshold():
    """Operation-level approval: triggered when risk ≥ threshold (0.6), otherwise allowed through."""
    svc = ApprovalService(config=ApprovalConfig(risk_threshold=0.6))
    high_op = Operation.create(Stage.CODE, "modify database schema", risk_level=OperationRiskLevel.EXTREME,
                               impact_scope=5, reversibility=0.1)
    low_op = Operation.create(Stage.CODE, "modify comments", risk_level=OperationRiskLevel.LOW,
                              impact_scope=1, reversibility=1.0)

    req_high = svc.check_operation_approval(high_op)
    req_low = svc.check_operation_approval(low_op)
    assert req_high is not None
    assert req_low is None
    assert req_high.mode == ApprovalMode.OPERATION


def test_stage_approval_only_in_review_deliver_by_default():
    """By default, stage-level approval triggers only at review and deliver."""
    svc = ApprovalService()
    assert svc.should_stage_approve(Stage.REVIEW)
    assert svc.should_stage_approve(Stage.DELIVER)
    assert not svc.should_stage_approve(Stage.PLAN)
    assert not svc.should_stage_approve(Stage.CODE)


def test_stage_approval_trigger_and_resolve():
    """Stage-level approval: high-risk review output triggers a request that can be approved/rejected; the record is kept."""
    svc = ApprovalService()
    out = make_review_output()
    request = svc.check_stage_approval(Stage.REVIEW, out,
                                       risk_items=[{"level": "high"}, {"level": "extreme"}],
                                       touches_critical=True)
    assert request is not None
    assert request.mode == ApprovalMode.STAGE
    assert request.approver == "code review lead"

    svc.resolve(request, ApprovalVerdict.APPROVED, "change follows standards")
    assert request.result == ApprovalVerdict.APPROVED
    assert request.resolved_at is not None
    assert len(svc.requests) == 1


def test_small_change_does_not_trigger_stage_approval():
    """Low-risk small changes do not trigger stage-level approval (risk < threshold)."""
    svc = ApprovalService()
    small = StageOutput(stage=Stage.REVIEW, files=["a.py"],
                        content={"risks": [{"level": "low"}]})
    request = svc.check_stage_approval(Stage.REVIEW, small, risk_items=[{"level": "low"}])
    assert request is None


def test_custom_approver_rejects():
    """A custom approver can reject; rejection triggers failure rollback."""
    class RejectingApprover(HumanApprover):
        def approve(self, request):
            from msef.engine.approval import ApprovalResult

            return ApprovalResult(request.id, ApprovalVerdict.REJECTED, "critical standards issue")

    svc = ApprovalService(approver=RejectingApprover())
    out = make_review_output()
    request = svc.check_stage_approval(Stage.REVIEW, out,
                                       risk_items=[{"level": "high"}], touches_critical=True)
    result = svc.resolve(request)
    assert result.verdict == ApprovalVerdict.REJECTED


def test_approval_record_callback_to_memory():
    """Approval records are persisted to the memory system (feedback loop)."""
    records = []

    def hook(req):
        records.append((req.id, req.result, req.approver))

    svc = ApprovalService(on_approval_record=hook)
    out = make_review_output()
    request = svc.check_stage_approval(Stage.REVIEW, out,
                                       risk_items=[{"level": "extreme"}], touches_critical=True)
    svc.decide(request)
    assert len(records) == 1
    assert records[0][0] == request.id
