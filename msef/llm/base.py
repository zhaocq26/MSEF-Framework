"""LLM backend abstraction.

MSEF funnels all calls to underlying model capabilities through the ``LLMBackend`` interface, decoupling framework logic (multi-model routing, temperature control, token accounting) from specific models. The model pool (DeepSeek-V4-Flash, Qwen2.5-Coder-32B, GPT-4o, Claude 3.5 Sonnet) connects to an enterprise API gateway via ``OpenAICompatibleBackend``; offline testing/demos use ``SimulatedBackend``.
"""
from __future__ import annotations

import abc
import json
import re
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    """LLM call exception."""


class LLMBackend(abc.ABC):
    """LLM call interface.

    Implementations must provide: model name, temperature (default 0.2), max_tokens cap, and a ``complete`` method.
    """

    name: str = "base"

    def __init__(self, temperature: float = 0.2, max_tokens: int = 8192) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens

    # -- Accounting ------------------------------------------------------------
    @abc.abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None, **kw: Any) -> str:
        """Return the model's output text for the given prompt."""

    # -- Convenience methods ---------------------------------------------------
    def complete_json(self, prompt: str, system: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
        """Request a JSON object from the model and parse it."""
        raw = self.complete(prompt, system=system, **kw)
        return parse_json_block(raw)

    def count_tokens(self, text: str) -> int:
        """Rough token count (Chinese by character, English by word; approximate)."""
        if not text:
            return 0
        chinese = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
        latin = len(re.findall(r"[A-Za-z0-9_]+", text))
        return chinese + latin

    def __repr__(self) -> str:  # pragma: no cover - debug use
        return f"<{type(self).__name__} name={self.name} temp={self.temperature}>"


def parse_json_block(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output (tolerates markdown code fences and surrounding noise)."""
    if not isinstance(text, str):
        raise LLMError(f"Model output is not text: {type(text)}")
    text = text.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find the first { to the last }
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError(f"No JSON object found in model output: {text[:200]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"JSON parsing failed: {exc}") from exc
