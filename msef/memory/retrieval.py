"""Automatic retrieval and mounting.

When the execution engine starts a new task, relevant memories are retrieved
based on task type and project features, then mounted into the current
context. The process has three steps:

1. Semantic matching: cosine similarity between the task query and entry
   vectors, filtering out entries below threshold θ_sim.
2. Priority ranking: composite score from usage frequency (freq), recency,
   scope level, and semantic similarity (sim); take Top-K.
3. Context mounting: format the results as prompt fragments appended to the
   input context.

Composite priority score:

    score = w_sim·sim + w_freq·freq_norm + w_rec·recency_norm + w_level·level_weight
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import torch

from msef.config import MemoryConfig
from msef.memory.entry import MemoryEntry
from msef.memory.storage import MemoryStore
from msef.types import MemoryScope


def recency_norm(last_access_at: float, now: float, window: float) -> float:
    """Recency normalization: more recent access yields a higher value."""
    if window <= 0:
        return 0.0
    age = max(0.0, now - last_access_at)
    return max(0.0, 1.0 - age / window)


def freq_norm(usage_count: int, max_usage: int) -> float:
    """Usage frequency normalization: entry usage count / max usage count across entries."""
    if max_usage <= 0:
        return 0.0
    return usage_count / max_usage


class MemoryRetriever:
    """Memory retriever."""

    def __init__(self, store: MemoryStore, config: Optional[MemoryConfig] = None) -> None:
        self.store = store
        self.config = config or MemoryConfig()

    # ------------------------------------------------------------------
    def semantic_match(self, query: str, candidates: Sequence[MemoryEntry],
                       threshold: float) -> List[Tuple[MemoryEntry, float]]:
        """Semantic matching: filter candidates by cosine similarity threshold."""
        if not candidates:
            return []
        q_vec = self.store.encode(query)            # [1, d]
        texts = [c.content for c in candidates]
        c_vec = self.store.encode(texts) if texts else torch.zeros((0, self.config.embedding_dim))
        if c_vec.shape[0] == 0:
            return []
        sims = self.store.encoder.cosine_similarity(q_vec, c_vec)[0]  # [K]
        hits = []
        for entry, sim in zip(candidates, sims.tolist()):
            if sim >= threshold:
                hits.append((entry, float(sim)))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def rank(self, query: str, hits: Sequence[Tuple[MemoryEntry, float]],
             top_k: int) -> List[Tuple[MemoryEntry, float, float]]:
        """Composite priority ranking, returning [(entry, similarity, priority)]."""
        if not hits:
            return []
        max_usage = max((e.usage_count for e, _ in hits), default=1)
        now = time.time()
        scored: List[Tuple[float, MemoryEntry, float]] = []
        for entry, sim in hits:
            freq = freq_norm(entry.usage_count, max_usage)
            rec = recency_norm(entry.last_access_at, now, self.config.recency_window_seconds)
            lvl = self.config.level_weights.get(entry.scope.value, 0.0)
            w = self.config.priority_weights
            priority = (
                w.get("sim", 0.4) * sim
                + w.get("freq", 0.2) * freq
                + w.get("recency", 0.2) * rec
                + w.get("level", 0.2) * lvl
            )
            scaled = priority * (0.8 if entry.downgraded else 1.0)  # decay for downgraded entries
            if entry.archived:
                continue  # archived entries skip regular retrieval (unless explicitly requested)
            scored.append((scaled, entry, sim))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(e, s, p) for p, e, s in scored[:top_k]]

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None,
                 scopes: Optional[List[MemoryScope]] = None,
                 touch_entries: bool = True) -> List[Tuple[MemoryEntry, float, float]]:
        """Full retrieval pipeline (semantic matching → priority ranking)."""
        top_k = top_k or self.config.top_k
        scopes = scopes or [MemoryScope.PROJECT, MemoryScope.TEAM, MemoryScope.PERSONAL]
        candidates = []
        for s in scopes:
            candidates.extend(self.store.entries(s))
        hits = self.semantic_match(query, candidates, self.config.similarity_threshold)
        ranked = self.rank(query, hits, top_k)
        if touch_entries:
            for entry, _, _ in ranked:
                self.store.touch(entry)
        return ranked

    # ------------------------------------------------------------------
    def mount(self, query: str, top_k: Optional[int] = None,
              scopes: Optional[List[MemoryScope]] = None) -> Tuple[str, List[Tuple[MemoryEntry, float, float]]]:
        """Retrieve and mount results into the context.

        Returns (mounted context text, retrieved entry list).
        """
        ranked = self.retrieve(query, top_k=top_k, scopes=scopes)
        if not ranked:
            return "", []
        sections = []
        for entry, sim, prio in ranked:
            level = entry.scope.label
            sections.append(
                f"[{level} memory · category:{entry.category}] (similarity={sim:.2f}, priority={prio:.2f})\n"
                f"{entry.content}"
            )
        mounted = "\n\n".join(sections)
        return mounted, ranked
