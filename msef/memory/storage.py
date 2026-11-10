"""Memory entry storage.

The three tiers use different carriers (project memory file / team command
library / user config) with correspondingly different formats. This module
provides a unified in-memory + JSON persistence store mapping to the three
carriers:

- Project tier: project memory file maintained alongside the codebase
  (``<root>/<MSEF_PROJECT_MEMORY_FILE>``)
- Team tier: shared team command library directory (``<team_dir>/commands.json``)
- Personal tier: local user config (``<home>/.msef/user_config.json``)

Semantic vectors are stored as PyTorch tensors (persisted as 1-D float32
lists), with metadata kept to support retrieval ranking and lifecycle
management.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import torch

from msef.memory.encoder import SemanticEncoder
from msef.memory.entry import MemoryEntry
from msef.types import MemoryScope


def _vector_to_list(vec) -> List[float]:
    if vec is None:
        return []
    if isinstance(vec, torch.Tensor):
        return vec.detach().cpu().reshape(-1).tolist()
    return [float(x) for x in vec]


def _squeeze_vector(vec) -> torch.Tensor:
    """Squeeze an [1, d] tensor to [d], unifying the 1-D vector representation."""
    if isinstance(vec, torch.Tensor) and vec.dim() > 1:
        return vec.reshape(-1)
    return vec


def _list_to_tensor(lst: List[float]) -> Optional[torch.Tensor]:
    if not lst:
        return None
    return torch.tensor(lst, dtype=torch.float32)


class MemoryStore:
    """Three-tier memory store (thread-safe).

    Args:
        encoder: Semantic encoder.
        project_root: Project root (location of the project memory file).
        team_dir: Team command library directory.
        user_home: User config directory.
        persist: Enable JSON persistence (default True).
        project_memory_file: Project memory file name.
    """

    DEFAULT_PROJECT_FILE = "MSEF_PROJECT_MEMORY.json"
    DEFAULT_TEAM_FILE = "commands.json"
    DEFAULT_USER_FILE = "user_config.json"

    def __init__(
        self,
        encoder: Optional[SemanticEncoder] = None,
        project_root: str = ".",
        team_dir: str = "",
        user_home: str = "",
        persist: bool = True,
        project_memory_file: str = DEFAULT_PROJECT_FILE,
    ) -> None:
        self.encoder = encoder or SemanticEncoder()
        self.project_root = Path(project_root)
        self.team_dir = Path(team_dir) if team_dir else self.project_root / ".msef-team"
        self.user_home = Path(user_home) if user_home else Path.home() / ".msef"
        self.persist = persist
        self.project_memory_file = project_memory_file
        self._lock = threading.RLock()
        # in-memory index: scope -> {entry_id: MemoryEntry}
        self._entries: Dict[str, Dict[str, MemoryEntry]] = {
            "project": {},
            "team": {},
            "personal": {},
        }
        self._scope_dir = {
            MemoryScope.PROJECT: self.project_root,
            MemoryScope.TEAM: self.team_dir,
            MemoryScope.PERSONAL: self.user_home,
        }
        self._scope_file = {
            MemoryScope.PROJECT: project_memory_file,
            MemoryScope.TEAM: self.DEFAULT_TEAM_FILE,
            MemoryScope.PERSONAL: self.DEFAULT_USER_FILE,
        }
        self._load_all()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, content: str, scope: MemoryScope, category: str = "general",
            source_type: str = "manual", entry_id: Optional[str] = None,
            vector: Optional[torch.Tensor] = None) -> MemoryEntry:
        """Add a new memory entry (with vector encoding; vector stored as 1-D)."""
        with self._lock:
            vec = vector if vector is not None else self.encode(content)
            entry = MemoryEntry(
                id=entry_id or uuid.uuid4().hex,
                content=content,
                scope=scope,
                vector=_squeeze_vector(vec),
                category=category,
                source_type=source_type,
                created_at=time.time(),
                last_access_at=time.time(),
                usage_count=0,
            )
            self._entries[scope.value][entry.id] = entry
            self._save(scope)
            return entry

    def update(self, entry: MemoryEntry, new_content: Optional[str] = None,
               touch: bool = True) -> None:
        """Update entry content and re-encode; refresh last access time and usage count."""
        with self._lock:
            if new_content is not None and new_content != entry.content:
                # versioning: keep the content diff before/after each update
                entry.versions.append({
                    "before": entry.content,
                    "after": new_content,
                    "timestamp": time.time(),
                    "reason": "content_update",
                })
                entry.content = new_content
                entry.vector = _squeeze_vector(self.encode(new_content))
            if touch:
                entry.last_access_at = time.time()
                entry.usage_count += 1
            self._save(entry.scope)

    def touch(self, entry: MemoryEntry) -> None:
        """Mark an access (called on retrieval hits)."""
        self.update(entry, touch=True)

    def delete(self, entry_id: str, scope: MemoryScope) -> Optional[MemoryEntry]:
        with self._lock:
            entry = self._entries[scope.value].pop(entry_id, None)
            if entry:
                self._save(scope)
            return entry

    def get(self, entry_id: str, scope: MemoryScope) -> Optional[MemoryEntry]:
        with self._lock:
            return self._entries[scope.value].get(entry_id)

    def entries(self, scope: Optional[MemoryScope] = None) -> List[MemoryEntry]:
        with self._lock:
            if scope is None:
                out = []
                for s in ("project", "team", "personal"):
                    out.extend(self._entries[s].values())
                return out
            return list(self._entries[scope.value].values())

    def count(self, scope: Optional[MemoryScope] = None) -> int:
        return len(self.entries(scope))

    # ------------------------------------------------------------------
    # Encoding and similarity
    # ------------------------------------------------------------------
    def encode(self, contents) -> torch.Tensor:
        return self.encoder.encode(contents)

    def scope_dir(self, scope: MemoryScope) -> Path:
        return self._scope_dir[scope]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _path_for(self, scope: MemoryScope) -> Path:
        return self._scope_dir[scope] / self._scope_file[scope]

    def _save(self, scope: MemoryScope) -> None:
        if not self.persist:
            return
        path = self._path_for(scope)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = [self._entry_to_dict(e) for e in self._entries[scope.value].values()]
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except OSError:  # pragma: no cover - persistence failure must not block the main flow
            pass

    def _load_all(self) -> None:
        if not self.persist:
            return
        for scope in (MemoryScope.PROJECT, MemoryScope.TEAM, MemoryScope.PERSONAL):
            path = self._path_for(scope)
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for d in data:
                    e = self._entry_from_dict(d)
                    self._entries[scope.value][e.id] = e
            except (OSError, json.JSONDecodeError):  # pragma: no cover
                pass

    def _entry_to_dict(self, e: MemoryEntry) -> Dict:
        return {
            "id": e.id,
            "content": e.content,
            "scope": e.scope.value,
            "vector": _vector_to_list(e.vector),
            "category": e.category,
            "source_type": e.source_type,
            "created_at": e.created_at,
            "last_access_at": e.last_access_at,
            "usage_count": e.usage_count,
            "consistency": e.consistency,
            "archived": e.archived,
            "downgraded": e.downgraded,
            "versions": e.versions,
        }

    def _entry_from_dict(self, d: Dict) -> MemoryEntry:
        e = MemoryEntry(
            id=d.get("id", uuid.uuid4().hex),
            content=d.get("content", ""),
            scope=MemoryScope(d.get("scope", "project")),
            vector=_list_to_tensor(d.get("vector") or []),
            category=d.get("category", "general"),
            source_type=d.get("source_type", "manual"),
            created_at=d.get("created_at", time.time()),
            last_access_at=d.get("last_access_at", time.time()),
            usage_count=d.get("usage_count", 0),
            consistency=d.get("consistency", 1.0),
            archived=d.get("archived", False),
            downgraded=d.get("downgraded", False),
            versions=d.get("versions", []),
        )
        return e