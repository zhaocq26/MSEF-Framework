"""Multi-model routing.

Selects the most suitable model from the candidate pool for a given task type/stage:

    m* = argmax_m S(m, ty)

where S is the model scoring function. The score matrix (LLM capability prior) can be learned/calibrated by the collaborative optimization mechanism; local models (deepseek-v4-flash / qwen2.5-coder-32b) handle high-intensity baseline code generation, while remote models (gpt-4o / claude-3.5-sonnet) handle high-level logic planning and test review (token control strategy).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from msef.config import AdaptationConfig
from msef.llm.base import LLMBackend
from msef.types import Stage, Task


class ModelRouter:
    """Multi-model router.

    Args:
        config: Adaptation layer configuration (candidate model pool and initial score matrix).
        backends: Model name → LLM backend instance mapping; defaults to returning a "virtual" backend name.
        score_override: Optional score overrides (for learning/experiments).
    """

    def __init__(self, config: Optional[AdaptationConfig] = None,
                 backends: Optional[Dict[str, LLMBackend]] = None,
                 score_override: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self.config = config or AdaptationConfig()
        self.backends = backends or {}
        self._scores: Dict[str, Dict[str, float]] = dict(self.config.model_scores)
        if score_override:
            for stage_scores in self._scores.values():
                pass
            for stage, sc in score_override.items():
                self._scores.setdefault(stage, {}).update(sc)

    # ------------------------------------------------------------------
    def score(self, stage: Stage, model: str) -> float:
        return self._scores.get(stage.value, {}).get(model, 0.0)

    def route(self, stage: Stage, task: Optional[Task] = None) -> str:
        """Select the best model name for the given stage."""
        candidates = self.config.models
        if not candidates:
            raise ValueError("Candidate model pool is empty")
        best = max(candidates, key=lambda m: self.score(stage, m))
        return best

    def backend_for(self, model: str) -> LLMBackend:
        if model not in self.backends:
            raise KeyError(f"Unregistered model backend: {model}; registered: {list(self.backends)}")
        return self.backends[model]

    # -- Collaborative optimization interface -----------------------------------
    def update_scores(self, grad: Dict[str, Dict[str, float]]) -> None:
        """Update the score matrix from collaborative optimization feedback (adaptation-layer results feed back into the execution engine)."""
        for stage, sc in grad.items():
            for model, delta in sc.items():
                cur = self._scores.setdefault(stage, {}).setdefault(model, 0.0)
                self._scores[stage][model] = min(1.0, max(0.0, cur + delta))

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {k: dict(v) for k, v in self._scores.items()}
