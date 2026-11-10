"""Memory update and lifecycle management.

The memory system is not a static store; it keeps updating as tasks execute.
This module implements:

1. Writing new sources (stage outputs / manual approvals / failure rollbacks /
   manual maintenance) with dedup-and-merge (update-or-add decided by semantic
   similarity).
2. Promotion and demotion across scopes (project ↔ team ↔ personal).
3. Lifecycle management: activity scoring (frequency / recency / consistency)
   plus downgrade, archive, and eliminate actions.
4. Versioning (updates keep history, enabling rollback) and conflict
   arbitration.

Activity score = w1·freq + w2·recency + w3·consistency.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch

from msef.config import MemoryConfig
from msef.memory.entry import MemoryEntry
from msef.memory.retrieval import freq_norm, recency_norm
from msef.memory.storage import MemoryStore
from msef.types import MemoryScope

logger = logging.getLogger(__name__)


class MemoryLifecycle:
    """Helper container for one entry's lifecycle state (activity, downgrade/archive/eliminate flags)."""

    def __init__(self, entry: MemoryEntry, activity: float) -> None:
        self.entry = entry
        self.activity = activity
        self.action: Optional[str] = None  # none | downgrade | archive | eliminate


class MemoryLifecycleManager:
    """Memory lifecycle manager.

    Args:
        store: The memory store.
        config: Memory config (thresholds, weights, time windows).
    """

    def __init__(self, store: MemoryStore, config: Optional[MemoryConfig] = None) -> None:
        self.store = store
        self.config = config or MemoryConfig()

    # ------------------------------------------------------------------
    # Activity
    # ------------------------------------------------------------------
    def activity_score(self, entry: MemoryEntry, entries_same_scope: List[MemoryEntry]) -> float:
        """Activity score for an entry."""
        now = time.time()
        max_usage = max((e.usage_count for e in entries_same_scope), default=1)
        freq = freq_norm(entry.usage_count, max_usage)
        rec = recency_norm(entry.last_access_at, now, self.config.recency_window_seconds)
        w = self.config.activity_weights
        return (
            w.get("freq", 0.4) * freq
            + w.get("recency", 0.3) * rec
            + w.get("consistency", 0.3) * entry.consistency
        )

    def consistency_gap(self, entry: MemoryEntry, entries_same_scope: List[MemoryEntry]) -> float:
        """Semantic consistency: mean similarity to other entries in the same scope."""
        peers = [e for e in entries_same_scope if e.id != entry.id and e.vector is not None]
        if not peers or entry.vector is None:
            return 1.0
        v = entry.vector.unsqueeze(0)
        vs = torch.stack([e.vector for e in peers])
        sims = self.store.encoder.cosine_similarity(v, vs)[0]
        return float(sims.mean())

    def evaluate(self, scope: MemoryScope) -> List[MemoryLifecycle]:
        """Evaluate lifecycle actions for all entries of a scope."""
        entries = self.store.entries(scope)
        if not entries:
            return []
        activities = [self.activity_score(e, entries) for e in entries]
        acts = sorted(activities)
        # thresholds: downgrade below median, archive below lower quartile, eliminate near minimum
        median = _percentile(acts, 0.5)
        q1 = _percentile(acts, 0.25)
        mini = _percentile(acts, 0.05)
        thresholds = self._thresholds(median, q1, mini)
        out = []
        for e, a in zip(entries, activities):
            lc = MemoryLifecycle(e, a)
            if a <= thresholds["eliminate"]:
                lc.action = "eliminate"
            elif a <= thresholds["archive"]:
                lc.action = "archive"
            elif a <= thresholds["downgrade"]:
                lc.action = "downgrade"
            out.append(lc)
        return out

    def _thresholds(self, median: float, q1: float, mini: float) -> Dict[str, float]:
        # recommended threshold distribution, with a small safety margin kept
        return {
            "downgrade": median * self.config.downgrade_ratio + median * 0.5,
            "archive": q1 + (median - q1) * 0.35,
            "eliminate": mini,
        }

    def apply(self, scope: MemoryScope) -> Dict[str, int]:
        """Apply lifecycle actions, returning per-action counts."""
        stats = {"downgrade": 0, "archive": 0, "eliminate": 0}
        for lc in self.evaluate(scope):
            e = lc.entry
            if lc.action == "downgrade" and not e.downgraded:
                e.downgraded = True
                stats["downgrade"] += 1
            elif lc.action == "archive" and not e.archived:
                e.archived = True
                e.downgraded = True
                stats["archive"] += 1
            elif lc.action == "eliminate":
                if e.archived or e.downgraded or _is_superseded(e):
                    if self.store.delete(e.id, scope):
                        stats["eliminate"] += 1
        return stats

    # ------------------------------------------------------------------
    # Write and merge
    # ------------------------------------------------------------------
    def write(self, content: str, scope: MemoryScope, category: str = "general",
              source_type: str = "stage_output",
              query_context: Optional[str] = None) -> Tuple[MemoryEntry, bool]:
        """Write new information; merge-update an existing entry when highly similar (≥ merge threshold), else add.

        Returns: (entry, merged_flag)
        """
        candidates = [e for e in self.store.entries(scope) if not e.archived]
        if candidates and self.store.encoder is not None:
            q = query_context or content
            q_vec = self.store.encode(q)
            c_vec = torch.stack([e.vector for e in candidates if e.vector is not None])
            if c_vec.shape[0] > 0 and q_vec.dim() == 2:
                sims = self.store.encoder.cosine_similarity(q_vec, c_vec)[0]
                best_idx = int(torch.argmax(sims).item())
                if float(sims[best_idx]) >= self.config.merge_similarity_threshold:
                    entry = candidates[best_idx]
                    self.store.update(entry, new_content=content, touch=True)
                    return entry, True
        entry = self.store.add(content, scope, category=category, source_type=source_type)
        return entry, False

    # ------------------------------------------------------------------
    # Promotion and demotion
    # ------------------------------------------------------------------
    def promote(self, entry_id: str, from_scope: MemoryScope, to_scope: MemoryScope) -> Optional[MemoryEntry]:
        """Promote an entry to a higher scope (personal→team, project→team)."""
        if from_scope == to_scope:
            return None
        entry = self.store.get(entry_id, from_scope)
        if entry is None:
            return None
        new_entry = self.store.add(
            entry.content, to_scope, category=entry.category,
            source_type=entry.source_type, vector=entry.vector,
        )
        self.store.delete(entry_id, from_scope)
        logger.info("Memory promotion: %s(%s) -> %s(%s)", entry_id, from_scope.value, new_entry.id, to_scope.value)
        return new_entry

    def demote(self, entry_id: str, from_scope: MemoryScope, to_scope: MemoryScope) -> Optional[MemoryEntry]:
        return self.promote(entry_id, from_scope, to_scope)

    # ------------------------------------------------------------------
    # Versioning and conflict arbitration
    # ------------------------------------------------------------------
    def rollback(self, entry: MemoryEntry, index: int = -1) -> bool:
        """Roll back to a historical version."""
        if not entry.versions:
            return False
        ver = entry.versions[index]
        before = ver.get("before", entry.content)
        self.store.update(entry, new_content=before)
        return True

    def resolve_conflict(self, a: MemoryEntry, b: MemoryEntry) -> MemoryEntry:
        """Conflict arbitration: source trust → update time → manual review (recorded as fallback)."""
        # simplified: trust = scope weight (project > team > personal); same scope takes latest update
        order = {"project": 0, "team": 1, "personal": 2}
        wa, wb = order[a.scope.value], order[b.scope.value]
        if wa != wb:
            return a if wa < wb else b
        return a if a.last_access_at >= b.last_access_at else b


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = int((len(vs) - 1) * q)
    return vs[k]


def _is_superseded(entry: MemoryEntry) -> bool:
    """Whether the entry was superseded by a newer version (simplified: has version history and content was overwritten)."""
    return len(entry.versions) > 1
