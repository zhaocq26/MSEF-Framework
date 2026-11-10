"""Context pre-compression.

Real-world code and documentation far exceed the model input window, so pre-compression selects the parts of the raw context most relevant to the current task and trims them to a target length:

    C' = ψ(C; L_max)

The strategy is "rank-and-truncate with task-sensitive weighting": each fragment's size (token count) is extracted, each fragment is scored by its relevance to the task description and affected files, and fragments are accumulated by descending score up to the target length. All ranking uses deterministic rules (task keyword hits, affected-file hits, code-context hits) rather than model judgment, guaranteeing reproducibility.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from msef.config import AdaptationConfig
from msef.types import Task
from msef.llm.base import LLMBackend


class ContextCompressor:
    """Context pre-compressor.

    Args:
        config: Adaptation layer configuration (target length, etc.).
        llm: LLM backend used for token estimation (optional; falls back to heuristic estimation).
    """

    def __init__(self, config: Optional[AdaptationConfig] = None,
                 llm: Optional[LLMBackend] = None) -> None:
        self.config = config or AdaptationConfig()
        self.llm = llm

    # ------------------------------------------------------------------
    def estimate_tokens(self, text: str) -> int:
        if self.llm is not None:
            return self.llm.count_tokens(text)
        # Heuristic: mixed CN/EN estimate, 1 token ≈ 1 Chinese char / word cluster
        cn = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin = len(re.findall(r"[A-Za-z0-9_]+", text))
        return cn + latin

    def score_fragment(self, fragment: str, task: Task, included_files: Sequence[str]) -> float:
        """Score a context fragment (higher means more worth keeping)."""
        score = 0.0
        low = fragment.lower()
        # 1. Match task title/description keywords
        for kw in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", task.title + " " + task.description):
            if kw.lower() in low:
                score += 1.5
        # 2. Appears in the affected-files list (file header fragments usually start with a path)
        for f in included_files:
            if f.lower().split("/")[-1] in low or f.lower() in low:
                score += 3.0
        # 3. Code/Chinese keyword weight
        if re.search(r"def |class |import ", fragment):
            score += 1.0
        score += min(1.0, len(fragment) / 400.0)  # Small information-content weight
        return score

    def compress(self, contexts: Sequence[str], task: Task,
                 target_tokens: Optional[int] = None, included_files: Sequence[str] = ()) -> str:
        """Compress a set of context fragments into a concatenated compressed context.

        Args:
            contexts: Raw context fragments (e.g. code text per file).
            task: The current task.
            target_tokens: Target token length; defaults to the configured value (32k).
            included_files: Task affected-file list (used for scoring).

        Returns:
            The concatenated compressed context (bounded in size).
        """
        target = target_tokens or self.config.context_target_tokens
        if not contexts:
            return ""
        scored: List[Tuple[float, str]] = []
        for frag in contexts:
            if not frag or not frag.strip():
                continue
            t = self.estimate_tokens(frag)
            if t <= 0:
                continue
            score = self.score_fragment(frag, task, included_files) / max(1.0, t) * 1000.0
            scored.append((score, frag))
        scored.sort(key=lambda x: x[0], reverse=True)

        budget = target
        kept: List[str] = []
        used = 0
        for score, frag in scored:
            t = self.estimate_tokens(frag)
            if used + t > budget and kept:
                continue
            kept.append(frag)
            used += t
            if used >= budget:
                break
        return "\n" + "\n".join(kept)[: len(kept) and used] if kept else ""