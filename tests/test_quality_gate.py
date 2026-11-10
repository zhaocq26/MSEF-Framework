"""Quality gate tests (per-stage evaluation dimensions and gate decisions)."""
import pytest

from msef.config import DEFAULT_GATE_DIMENSIONS
from msef.engine.quality_gate import (
    QualityGateService,
    score_compile_pass_rate,
    score_impact_coverage,
    score_plan_completeness,
    score_potential_risk,
    score_regression_result,
    score_report_completeness,
    score_risk_sufficiency,
    score_static_alert_count,
    score_syntax_correctness,
    score_unit_pass_rate,
)
from msef.types import QualityReport, Stage, StageOutput


def make_output(stage: Stage, **content) -> StageOutput:
    return StageOutput(stage=stage, content=content,
                       files=content.get("files") or [])


def test_plan_gate_pass():
    """Plan stage: complete plan / full impact coverage / sufficient risks → pass."""
    svc = QualityGateService()
    ctx = {
        "required_plan_elements": ["scope", "dependencies", "sequence", "risks", "rollback"],
        "affected_files": ["a.py", "b.py"],
        "required_risks": ["database change risk"],
    }
    out = make_output(
        Stage.PLAN,
        plan_elements=["scope", "dependencies", "sequence", "risks", "rollback"],
        files=["a.py", "b.py"],
        risks=["database change risk"],
    )
    report = svc.evaluate(Stage.PLAN, out, ctx)
    assert report.passed
    assert report.weighted_score >= 0.7
    assert {d.name for d in report.dimension_scores} == set(DEFAULT_GATE_DIMENSIONS["plan"])


def test_plan_gate_fail_when_incomplete():
    """Plan stage: incomplete plan → gate fails; dimension scores point to improvement areas."""
    svc = QualityGateService()
    ctx = {
        "required_plan_elements": ["scope", "dependencies", "sequence", "risks", "rollback"],
        "affected_files": ["a.py", "b.py"],
        "required_risks": ["database change risk"],
    }
    out = make_output(
        Stage.PLAN,
        plan_elements=[],
        files=[],
        risks=[],
    )
    report = svc.evaluate(Stage.PLAN, out, ctx)
    assert not report.passed
    assert report.score_map()["plan_completeness"] == 0.0
    assert report.score_map()["impact_coverage"] == 0.0


def test_code_gate_dimensions():
    """Code stage: syntax/compile/spec three dimensions."""
    svc = QualityGateService()
    out = make_output(
        Stage.CODE,
        files=["a.py"],
        syntax_ok=["a.py"],
        compile_ok=["a.py"],
        spec_ok=1.0,
    )
    report = svc.evaluate(Stage.CODE, out, {})
    assert report.passed
    assert report.score_map()["syntax_correctness"] == 1.0
    assert report.score_map()["compile_pass_rate"] == 1.0
    assert report.score_map()["spec_compliance"] == 1.0


def test_code_gate_weighted_direction():
    """Code stage: syntax correctness has the highest weight (0.5) → syntax failure drags down the total."""
    from msef.engine.quality_gate import QualityGate

    gate = QualityGate(
        Stage.CODE,
        DEFAULT_GATE_DIMENSIONS["code"],
        threshold=0.6,
    )
    out = make_output(Stage.CODE, files=["a.py"], syntax_ok=False, compile_ok=True, spec_ok=1.0)
    report = gate.evaluate(out, {})
    # 0.5*0 + 0.3*1 + 0.2*1 = 0.5 < 0.6
    assert not report.passed
    assert 0.49 < report.weighted_score < 0.51


def test_test_gate_regression_direction():
    """Test stage: regression result = 1 - new defect introduction rate (higher is better)."""
    out = make_output(Stage.TEST, unit_pass=0.9, integration_pass=0.9, new_defects=0.1)
    assert score_regression_result(out, {}) == pytest.approx(0.9)
    out2 = make_output(Stage.TEST, unit_pass=0.9, integration_pass=0.9, new_defects=0.9)
    assert score_regression_result(out2, {}) == pytest.approx(0.1)


def test_review_risk_score():
    """Review stage: higher risk → lower potential-risk score (normalized so higher is better)."""
    low_risk = make_output(
        Stage.REVIEW,
        risk_items=[{"level": "low"}],
        alerts=1,
        spec_ok=1.0,
    )
    high_risk = make_output(
        Stage.REVIEW,
        risk_items=[{"level": "extreme"}, {"level": "high"}],
        alerts=20,
        spec_ok=0.5,
    )
    r_low = score_potential_risk(low_risk, {})
    r_high = score_potential_risk(high_risk, {})
    assert r_low > r_high
    assert score_static_alert_count(high_risk, {}) < score_static_alert_count(low_risk, {})


def test_deliver_gate_proxies():
    """Deliver stage: report fields act as deliverable proxies → pass."""
    svc = QualityGateService()
    out = make_output(
        Stage.DELIVER,
        summary="summary", changes=["c"], risks=["r"],
        tests=["t"], rollback="rb", report_complete=True, risks_flagged=True,
    )
    report = svc.evaluate(Stage.DELIVER, out, {})
    assert report.passed
    assert report.score_map()["report_completeness"] == 1.0
    assert report.score_map()["deliverable_completeness"] == 1.0


def test_gate_report_structure():
    """Gate result includes dimension scores, weighted score, threshold, pass flag, and failing dimensions."""
    svc = QualityGateService()
    out = make_output(Stage.TEST, unit_pass=0.5, integration_pass=0.5, new_defects=0.2)
    report: QualityReport = svc.evaluate(Stage.TEST, out, {})
    assert isinstance(report, QualityReport)
    assert len(report.dimension_scores) == 3
    assert not report.passed
    assert report.failing_dimensions


def test_gate_threshold_override():
    """The test-stage threshold of 0.85 is the strictest."""
    svc = QualityGateService()
    assert svc.gate(Stage.TEST).threshold == 0.85
    assert svc.gate(Stage.PLAN).threshold == 0.7
