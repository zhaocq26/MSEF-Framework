"""OpenAI-compatible LLM backend.

Connects to models on an enterprise API gateway such as DeepSeek-V4-Flash, Qwen2.5-Coder-32B, GPT-4o, and Claude 3.5 Sonnet. base_url / api_key / model are injected via environment variables or the constructor. The openai SDK is an optional dependency (``pip install msef[llm]``).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from msef.llm.base import LLMBackend


class OpenAICompatibleBackend(LLMBackend):
    """Model backend for OpenAI-compatible APIs.

    Args:
        model: Model name (e.g. "gpt-4o" / "claude-3.5-sonnet" / "deepseek-v4-flash").
        base_url: API gateway URL; defaults to the ``MSEF_LLM_BASE_URL`` environment variable.
        api_key: API key; defaults to ``MSEF_LLM_API_KEY``.
        temperature: Sampling temperature (default 0.2).
    """

    name = "openai-compatible"

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 temperature: float = 0.2, max_tokens: int = 8192, timeout: float = 300.0) -> None:
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model = model
        self.base_url = base_url or os.environ.get("MSEF_LLM_BASE_URL", "")
        self.api_key = api_key or os.environ.get("MSEF_LLM_API_KEY", "")
        self.timeout = timeout
        if not self.base_url or not self.api_key:
            raise ValueError("OpenAICompatibleBackend requires base_url and api_key (or MSEF_LLM_BASE_URL/MSEF_LLM_API_KEY)")

    def complete(self, prompt: str, system: Optional[str] = None, **kw: Any) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - missing dependency
            raise RuntimeError("Install the openai SDK: pip install 'msef[llm]'") from exc

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        messages: list = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=kw.get("max_tokens", self.max_tokens),
        )
        return resp.choices[0].message.content or ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<OpenAICompatibleBackend model={self.model} base_url={self.base_url}>"
