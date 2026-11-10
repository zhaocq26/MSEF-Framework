"""Simulated LLM backend (offline testing and demos).

This backend is not a real model: it returns deterministic structured outputs per task scenario and stage, enabling end-to-end verification of MSEF's multi-stage execution engine, quality gates, human approval, failure rollback, and checkpoint resume logic without any API dependency. Use ``OpenAICompatibleBackend`` to connect real models.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, Optional

from msef.llm.base import LLMBackend


class SimulatedBackend(LLMBackend):
    """Deterministic script-based simulation backend.

    Args:
        profile: Simulation profile, e.g.::

            {
                "plan":   {"plan_completeness": 0.9, "impact_coverage": 0.8, "risk_sufficiency": 0.9,
                           "files": ["a.py", "b.py"], "risks": ["Risk A"]},
                "code":   {"syntax_ok": ["a.py"], "compile_ok": ["a.py"], "spec_ok": 0.9},
                "test":   {"unit_pass": 0.9, "integration_pass": 0.92, "new_defects": 0.03},
                "review": {"spec_ok": 0.9, "risk_items": [{"level": "low"}], "alerts": 2},
                "deliver": {"report_complete": True, "risks_flagged": True},
            }

        The backend produces per-stage outputs from this profile; after the profile values pass through quality-gate computation, they can drive both pass and fail execution paths.
    """

    name = "simulated"

    def __init__(self, profile: Optional[Dict[str, Any]] = None, temperature: float = 0.2,
                 quality_noise: float = 0.0, seed: int = 42,
                 retry_profile: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(temperature=temperature)
        self.profile = profile or {}
        self.retry_profile = retry_profile or {}
        self.quality_noise = quality_noise
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def complete(self, prompt: str, system: Optional[str] = None, **kw: Any) -> str:
        # Identify the current stage and task requirements from the prompt
        stage = self._detect_stage(prompt)
        scenario = self.profile.get(stage, {})
        payload = self._render_stage(stage, scenario, prompt)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    @staticmethod
    def _detect_stage(prompt: str) -> str:
        """Detect the stage from the prompt text.

        Stage instructions are appended at the end of the prompt (see Agent._build_prompt), so the last stage marker is taken as the current stage, avoiding interference from stage-like words in mounted memory or prior outputs.
        """
        p = prompt.lower()
        markers = {
            "stage:plan": "plan", "stage:code": "code", "stage:test": "test",
            "stage:review": "review", "stage:deliver": "deliver",
            "planning": "plan", "coding": "code", "testing": "test",
            "review": "review", "delivery": "deliver",
        }
        best_idx, best_stage = -1, "plan"
        for marker, stage in markers.items():
            idx = p.rfind(marker)
            if idx > best_idx:
                best_idx, best_stage = idx, stage
        return best_stage

    def _jitter(self, value: float) -> float:
        if not self.quality_noise or not isinstance(value, (int, float)):
            return value
        noise = self._rng.uniform(-self.quality_noise, self.quality_noise)
        return max(0.0, min(1.0, value + noise))

    @staticmethod
    def _retry_boost(prompt: str) -> float:
        """When the prompt contains failure-rollback suggestions, simulate the quality improvement of an agent that fixes output per the suggestions."""
        if any(kw in prompt.lower() for kw in ("rollback suggestion", "quality gate failed", "retry", "rejection")):
            return 0.18
        return 0.0

    def _boost(self, value: float, prompt: str) -> float:
        if not isinstance(value, (int, float)):
            return value
        return max(0.0, min(1.0, value + self._retry_boost(prompt)))

    def _render_stage(self, stage: str, scenario: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        retry = self._retry_boost(prompt) > 0.0
        s = dict(scenario)
        if retry:
            # After failure-rollback feedback, the simulated agent revises output per the suggestions (configurable via retry_profile)
            s.update(self.retry_profile.get(stage, {}))
        out: Dict[str, Any] = {"stage": stage}
        if stage == "plan":
            out["plan_completeness"] = self._boost(self._jitter(s.get("plan_completeness", 0.85)), prompt)
            out["impact_coverage"] = self._boost(self._jitter(s.get("impact_coverage", 0.8)), prompt)
            out["risk_sufficiency"] = self._boost(self._jitter(s.get("risk_sufficiency", 0.85)), prompt)
            out["files"] = s.get("files") or self._extract_files(prompt)
            out["risks"] = s.get("risks", [])
            out["operations"] = s.get("operations", [])
            if "plan_elements" in s:
                out["plan_elements"] = s["plan_elements"]   # An explicit empty list means an incomplete plan
            else:
                out["plan_elements"] = [
                    "scope", "dependencies", "sequence", "risks", "rollback",
                ]
            out["plan"] = self._plan_text(prompt)
        elif stage == "code":
            out["syntax_ok"] = s.get("syntax_ok", True)
            out["compile_ok"] = s.get("compile_ok", True)
            out["spec_ok"] = self._boost(self._jitter(s.get("spec_ok", 0.9)), prompt)
            out["files"] = s.get("files") or self._extract_files(prompt)
            out["changes"] = self._changes_text(prompt)
            out["operations"] = s.get("operations", [])
        elif stage == "test":
            out["unit_pass"] = self._boost(self._jitter(s.get("unit_pass", 0.9)), prompt)
            out["integration_pass"] = self._boost(self._jitter(s.get("integration_pass", 0.9)), prompt)
            out["new_defects"] = self._boost(self._jitter(s.get("new_defects", 0.03)), prompt)
        elif stage == "review":
            out["spec_ok"] = self._boost(self._jitter(s.get("spec_ok", 0.9)), prompt)
            out["risk_items"] = s.get("risk_items", [])
            out["alerts"] = int(s.get("alerts", 2))
            out["files"] = s.get("files") or self._extract_files(prompt)
        elif stage == "deliver":
            out["report_complete"] = s.get("report_complete", True)
            out["risks_flagged"] = s.get("risks_flagged", True)
            out["summary"] = s.get("summary", "Implemented and delivered the requested change")
            out["changes"] = s.get("changes", ["source changes", "test cases"])
            out["risks"] = s.get("risks", ["operational risk", "regression risk"])
            out["tests"] = s.get("tests", ["unit tests", "integration tests"])
            out["rollback"] = s.get("rollback", "Rollback with git revert")
            out["files"] = s.get("files") or self._extract_files(prompt)
            out["exact_match"] = float(s.get("exact_match", 0.0))
        return out

    @staticmethod
    def _plan_text(prompt: str) -> str:
        return (
            "Execution plan:\n"
            "1. Analyze impact and list affected files\n"
            "2. Implement changes in dependency order\n"
            "3. Write and run unit and integration tests\n"
            "4. Complete code review and static analysis\n"
            "5. Produce the change report and deliver"
        )

    @staticmethod
    def _extract_files(prompt: str) -> list:
        """Extract file paths from the prompt's subtask list (e.g. ``- [a.py, b.py] description``)."""
        import re

        found = []
        for m in re.finditer(r"\[([^\]]+)\]", prompt):
            for part in m.group(1).split(","):
                name = part.strip().strip("'\"")
                if name and (
                    "." in name
                    or name.startswith(("src", "test", "lib", "app"))
                ):
                    found.append(name)
        # Deduplicate while preserving order
        return list(dict.fromkeys(found))[:20]

    @staticmethod
    def _changes_text(prompt: str) -> str:
        return "Implemented the planned source changes and updated the relevant tests."
