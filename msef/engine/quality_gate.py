"""Quality gates.

A quality gate evaluates the previous stage's output at each stage boundary,
serving three purposes: gating (deciding whether the output is qualified),
direction (per-dimension scores point out improvement areas), and experience
filtering (gate results are written to the memory system).

The quality score function is defined per stage:

    Q_k(y_k) = Σ_j w_{k,j} · q_{k,j}(y_k)

All evaluation dimensions are produced by deterministic tools or computed from
structured records, not by subjective model judgment.

Per-stage dimensions:
- plan: plan completeness, impact scope identification (reachability analysis), risk assessment sufficiency
- code: syntax correctness (syntax checker), compile pass rate (compiler), spec compliance (spec checker)
- test: unit test pass rate, integration test pass rate, regression result (1 - new defect introduction rate)
- review: spec compliance, potential risk score (complexity/dependency/history defects), static analysis alert count
- deliver: change report completeness, risk item marking sufficiency, deliverable completeness
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from msef.config import Config
from msef.types import DimensionScore, QualityReport, Stage, StageOutput


# ---------------------------------------------------------------------------
# Per-dimension scoring functions
# ---------------------------------------------------------------------------
def ratio(numerator: float, denominator: float) -> float:
    """Normalized ratio (0/0 treated as 1)."""
    if denominator <= 0:
        return 1.0
    return max(0.0, min(1.0, numerator / denominator))


def score_plan_completeness(output: StageOutput, task_context: Dict[str, Any]) -> float:
    """Plan completeness: covered required elements / total required elements."""
    content = output.content
    required: set = {e for e in task_context.get("required_plan_elements", [])}
    covered = {e for e in content.get("plan_elements", [])} | set(content.get("plan", "").split("\n"))
    if not required:
        return 1.0
    matched = {e for e in required if any(e in c for c in covered)}
    return ratio(len(matched), len(required))


def score_impact_coverage(output: StageOutput, task_context: Dict[str, Any]) -> float:
    """Impact scope coverage: identified affected files / actual affected files (static reachability)."""
    actual = set(task_context.get("affected_files", output.files))
    identified = set(content_files(output)) | set(output.files)
    if not actual:
        return 1.0
    # Actual affected files come from reachability analysis along the call graph
    # from entry files (simulated here with task ground truth)
    return ratio(len(actual & identified), len(actual))


def content_files(output: StageOutput) -> List[str]:
    files: List[str] = list(output.files)
    for key in ("files", "changed_files", "affected_files"):
        v = output.content.get(key)
        if isinstance(v, (list, tuple)):
            files.extend(str(x) for x in v)
    return list(dict.fromkeys(files))


def score_risk_sufficiency(output: StageOutput, task_context: Dict[str, Any]) -> float:
    """Risk assessment sufficiency: marked risks / required risks."""
    required = set(task_context.get("required_risks", []))
    marked = set(output.content.get("risks", [])) | set(output.content.get("risk_flags", []))
    if not required:
        return 1.0
    return ratio(len(required & marked), len(required))


def score_syntax_correctness(output: StageOutput, _task_context: Dict[str, Any]) -> float:
    """Syntax correctness: files passing syntax check / total changed files (syntax checker output)."""
    files = content_files(output)
    if not files:
        return 0.0
    syn_ok = output.content.get("syntax_ok")
    if isinstance(syn_ok, list):
        ok = len(set(syn_ok) & set(files))
        return ratio(ok, len(files))
    if isinstance(syn_ok, bool):
        return 1.0 if syn_ok else 0.0
    return float(output.content.get("syntax_pass_rate", 1.0 if syn_ok is None else syn_ok))


def score_compile_pass_rate(output: StageOutput, _task_context: Dict[str, Any]) -> float:
    """Compile pass rate: modules compiling / total changed modules (compiler output)."""
    rate = output.content.get("compile_pass_rate")
    if rate is not None:
        return float(rate)
    comp = output.content.get("compile_ok")
    if isinstance(comp, bool):
        return 1.0 if comp else 0.0
    if isinstance(comp, list):
        files = content_files(output)
        return ratio(len(set(comp) & set(files)), max(1, len(files)))
    return 1.0 if comp is None else float(comp)


def score_spec_compliance(output: StageOutput, _task_context: Dict[str, Any]) -> float:
    """Spec compliance: ratio of changed lines conforming to coding spec (spec checker output)."""
    v = output.content.get("spec_ok")
    if v is None:
        v = output.content.get("spec_compliance", 1.0)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v)


def score_unit_pass_rate(output: StageOutput, _task_context: Dict[str, Any]) -> float:
    """Unit test pass rate: passing unit test cases / total executed."""
    return _pass_rate(output, "unit")


def score_integration_pass_rate(output: StageOutput, _task_context: Dict[str, Any]) -> float:
    """Integration test pass rate."""
    return _pass_rate(output, "integration")


def score_regression_result(output: StageOutput, _task_context: Dict[str, Any]) -> float:
    """Regression result: 1 - new defect introduction rate (higher is better)."""
    new_defects = output.content.get("new_defects")
    if new_defects is None:
        new_defects = output.content.get("regression_new_defect_rate", 0.0)
    return max(0.0, min(1.0, 1.0 - float(new_defects)))


def _pass_rate(output: StageOutput, kind: str) -> float:
    key_map = {
        "unit": ("unit_pass", "unit_pass_rate"),
        "integration": ("integration_pass", "integration_pass_rate"),
    }
    for key in key_map[kind]:
        if key in output.content:
            return float(output.content[key])
    return 0.0


def score_review_spec_compliance(output: StageOutput, _ctx: Dict[str, Any]) -> float:
    return score_spec_compliance(output, _ctx)


def score_potential_risk(output: StageOutput, _ctx: Dict[str, Any]) -> float:
    """Potential risk score (weighted Complexity/Dependency/History synthesis; lower raw risk is better).

    Returns a "higher is better" form: 1 - normalized risk value.
    """
    risk_items = output.content.get("risk_items") or []
    weighted = 0.0
    weight_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "extreme": 0.95}
    for item in risk_items:
        if isinstance(item, dict):
            lvl = item.get("level", "medium")
        else:
            lvl = str(item)
        weighted += weight_map.get(lvl, 0.5)
    n = max(1, len(risk_items))
    risk_avg = min(1.0, weighted / n)
    complexity = float(output.content.get("complexity", 0.3))
    deps = float(output.content.get("dependency_scope", 0.2))
    history = float(output.content.get("history_defect_density", 0.2))
    w = (0.4, 0.3, 0.3)
    p = w[0] * complexity + w[1] * deps + w[2] * history
    total = 0.5 * p + 0.5 * risk_avg
    return max(0.0, min(1.0, 1.0 - total))


def score_static_alert_count(output: StageOutput, _ctx: Dict[str, Any]) -> float:
    """Static analysis alert count: 1 if alerts within allowance, proportional deduction above it."""
    alerts = output.content.get("alerts")
    if alerts is None:
        alerts = len(output.content.get("static_alerts") or [])
    allowed = int(output.content.get("max_alerts", 5))
    if alerts <= allowed:
        return 1.0
    return max(0.0, 1.0 - (alerts - allowed) / 10.0)


def score_report_completeness(output: StageOutput, _ctx: Dict[str, Any]) -> float:
    """Change report completeness: ratio of required fields present in the report."""
    required_fields = ["summary", "changes", "risks", "tests", "rollback"]
    content = output.content
    present = 0
    for f in required_fields:
        v = content.get(f)
        if isinstance(v, str):
            if v.strip():
                present += 1
        elif isinstance(v, (list, dict)) and v:
            present += 1
        elif v is True:
            present += 1
    explicit = content.get("report_complete")
    if explicit is False:
        return 0.3
    return ratio(present, len(required_fields))


def score_risk_marking(output: StageOutput, _ctx: Dict[str, Any]) -> float:
    """Risk item marking sufficiency: flagged risks / known risks."""
    content = output.content
    if content.get("risks_flagged") is False:
        return 0.0
    known = _ctx.get("required_risks", [])
    flagged = content.get("risks") or content.get("risk_flags") or []
    if not known:
        return 1.0 if content.get("risks_flagged", True) else 0.0
    return ratio(len(set(known) & set(flagged)), len(set(known)))


def score_deliverable_completeness(output: StageOutput, _ctx: Dict[str, Any]) -> float:
    """Deliverable completeness: deliverables cover code changes, test report, review opinion, etc."""
    content = output.content
    deliverables = content.get("deliverables") or []
    required = ["code_changes", "test_report", "review_opinion", "change_report"]
    if not deliverables:
        # When deliverables are not declared explicitly, infer from report fields:
        # changes→code changes, tests→test report, risks/rollback→review opinion, summary→change report
        proxies = [
            ("code_changes", ["changes", "syntax_ok", "compile_ok"]),
            ("test_report", ["tests", "unit_pass", "integration_pass"]),
            ("review_opinion", ["risks", "rollback", "risk_items"]),
            ("change_report", ["summary", "report_complete"]),
        ]
        present = sum(
            1 for _, keys in proxies
            if any(content.get(k) is not None for k in keys)
        )
        return ratio(present, len(required))
    return ratio(len(set(deliverables) & set(required)), len(required))


# ---------------------------------------------------------------------------
# Stage → per-dimension scorer registry
# ---------------------------------------------------------------------------
STAGE_SCORERS: Dict[str, List[Callable[[StageOutput, Dict[str, Any]], float]]] = {
    "plan": [score_plan_completeness, score_impact_coverage, score_risk_sufficiency],
    "code": [score_syntax_correctness, score_compile_pass_rate, score_spec_compliance],
    "test": [score_unit_pass_rate, score_integration_pass_rate, score_regression_result],
    "review": [score_review_spec_compliance, score_potential_risk, score_static_alert_count],
    "deliver": [score_report_completeness, score_risk_marking, score_deliverable_completeness],
}


class QualityGate:
    """Per-stage quality gate: score each dimension → weight → compare with threshold."""

    def __init__(self, stage: Stage, dimensions: Dict[str, float],
                 threshold: float, scorers: Optional[List[Callable]] = None) -> None:
        self.stage = stage
        self.dimensions = dimensions          # dimension name -> weight
        self.threshold = threshold
        self.scorers = scorers or STAGE_SCORERS.get(stage.value, [])

    def evaluate(self, output: StageOutput, task_context: Optional[Dict[str, Any]] = None) -> QualityReport:
        ctx = task_context or {}
        scores: List[DimensionScore] = []
        names = list(self.dimensions.keys())
        for i, scorer in enumerate(self.scorers):
            name = names[i] if i < len(names) else f"dim{i}"
            raw = 0.0
            try:
                raw = scorer(output, ctx)
            except Exception:  # pragma: no cover - scorer exceptions must not block the gate
                raw = 0.0
            detail = {"function": scorer.__name__}
            scores.append(DimensionScore(name=name, score=max(0.0, min(1.0, raw)), detail=detail))
        weighted = sum(self.dimensions.get(d.name, 0.0) * d.score for d in scores)
        # Normalize weights (in case user-supplied weights are not normalized)
        total_w = sum(self.dimensions.values()) or 1.0
        weighted = weighted / total_w
        return QualityReport(
            stage=self.stage,
            dimension_scores=scores,
            weighted_score=round(weighted, 4),
            threshold=self.threshold,
            passed=weighted >= self.threshold,
        )


class QualityGateService:
    """Quality gate service: manages one gate instance per stage."""

    def __init__(self, config: Optional[Config] = None,
                 overrides: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self.config = config or Config()
        self._gates: Dict[Stage, QualityGate] = {}
        for stage in Stage.ordered():
            dims = dict(self.config.gate_dimensions.get(stage.value, {}))
            if overrides and overrides.get(stage.value):
                dims.update(overrides[stage.value])
            thr = self.config.gate_thresholds.get(stage.value, 0.7)
            self._gates[stage] = QualityGate(stage, dims, thr)

    def gate(self, stage: Stage) -> QualityGate:
        return self._gates[stage]

    def evaluate(self, stage: Stage, output: StageOutput,
                 task_context: Optional[Dict[str, Any]] = None) -> QualityReport:
        return self._gates[stage].evaluate(output, task_context)

    def evaluate_all(self, outputs: Dict[str, StageOutput]) -> Dict[str, QualityReport]:
        return {
            s.value: self.evaluate(Stage(s), out)
            for s, out in [(Stage(k), v) for k, v in outputs.items()]
        }