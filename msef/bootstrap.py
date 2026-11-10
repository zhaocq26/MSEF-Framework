"""MSEF system assembly (bootstrap).

Assembles the model adaptation layer, multi-stage execution engine, three-tier memory system, and collaborative optimization mechanism into a fully runnable system. Tests and examples build MSEF instances quickly through this module.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from msef.adaptation.compression import ContextCompressor
from msef.adaptation.decomposition import DecompositionPolicy, TaskReducer
from msef.adaptation.router import ModelRouter
from msef.agent.agent import Agent
from msef.config import Config
from msef.engine.approval import ApprovalService
from msef.engine.engine import MultiStageExecutionEngine
from msef.engine.quality_gate import QualityGateService
from msef.engine.rollback import CheckpointManager, RollbackManager
from msef.llm.base import LLMBackend
from msef.llm.simulated import SimulatedBackend
from msef.memory.retrieval import MemoryRetriever
from msef.memory.storage import MemoryStore
from msef.optimization.feedback import CollaborativeOptimizer

logger = logging.getLogger(__name__)


class MSEFSystem:
    """Container of the MSEF system: holds all components and provides a unified entry point.

    Attributes:
        config: Global configuration.
        store: Three-tier memory storage.
        retriever: Memory retriever.
        engine: Multi-stage execution engine.
        optimizer: Collaborative optimizer.
        backend: LLM backend (simulated by default).
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        backend: Optional[LLMBackend] = None,
        retry_profile: Optional[Dict[str, Any]] = None,
        approver=None,
        data_dir: Optional[str] = None,
        seed: int = 42,
        persist: bool = True,
    ) -> None:
        self.config = config or Config(seed=seed)
        if data_dir is None:
            data_dir = tempfile.mkdtemp(prefix="msef_")
        root = Path(data_dir)
        self.data_dir = root

        # Three-tier memory system
        self.store = MemoryStore(
            encoder=None,
            project_root=str(root / "project"),
            team_dir=str(root / "team"),
            user_home=str(root / "user"),
            persist=persist,
        )
        self.retriever = MemoryRetriever(self.store, self.config.memory)

        # Model adaptation layer
        self.policy = DecompositionPolicy(
            n_granularities=len(self.config.adaptation.granularity_levels),
            feature_dim=8,
            hidden_dim=self.config.optimization.hidden_dim,
            seed=seed,
        )
        self.reducer = TaskReducer(policy=self.policy, config=self.config.adaptation, seed=seed)
        self.compressor = ContextCompressor(self.config.adaptation)
        self.router = ModelRouter(self.config.adaptation)

        # LLM backend and agent
        self.backend = backend or SimulatedBackend(profile={}, retry_profile=retry_profile, seed=seed)
        self.agent = Agent(backend=self.backend)

        # Multi-stage execution engine
        approval_svc = ApprovalService(
            config=self.config.approval, approver=approver,
        )
        self.engine = MultiStageExecutionEngine(
            agent=self.agent,
            reducer=self.reducer,
            compressor=self.compressor,
            router=self.router,
            gates=QualityGateService(self.config),
            approvals=approval_svc,
            rollback=RollbackManager(max_retries=self.config.max_retries),
            checkpoint=CheckpointManager(enabled=self.config.checkpoint_enabled),
            store=self.store,
            retriever=self.retriever,
            config=self.config,
        )

        # Collaborative optimization mechanism
        self.optimizer = CollaborativeOptimizer(
            reducer=self.reducer,
            store=self.store,
            config=self.config.optimization,
        )

    # ------------------------------------------------------------------
    def run_task(self, task, contexts: Optional[List[str]] = None):
        """Run a single task and collect experience samples."""
        result = self.engine.run(task, contexts=contexts)
        self.optimizer.collect(result)
        return result

    def maybe_optimize(self, completed_tasks: int) -> bool:
        """Run collaborative optimization when the update cycle is reached (memory → adaptation layer feedback)."""
        return self.optimizer.maybe_update(completed_tasks)

    def seed_memory(self, project_entries=None, team_entries=None, personal_entries=None) -> None:
        """Pre-seed three-tier memory (project/team/personal) with agent cold-start knowledge."""
        from msef.types import MemoryScope

        for content, category in (project_entries or []):
            if not any(e.content == content for e in self.store.entries(MemoryScope.PROJECT)):
                self.store.add(content, MemoryScope.PROJECT, category=category, source_type="manual")
        for content, category in (team_entries or []):
            if not any(e.content == content for e in self.store.entries(MemoryScope.TEAM)):
                self.store.add(content, MemoryScope.TEAM, category=category, source_type="manual")
        for content, category in (personal_entries or []):
            if not any(e.content == content for e in self.store.entries(MemoryScope.PERSONAL)):
                self.store.add(content, MemoryScope.PERSONAL, category=category, source_type="manual")

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """System run summary (experiment metrics)."""
        m = self.engine.metrics
        return {
            "SPR": round(m.stage_pass_rate, 3),
            "RBR": round(m.rollback_rate, 3),
            "HIR": round(m.approvals / max(1, m.completed_tasks), 3),
            "completed_tasks": m.completed_tasks,
            "average_duration_seconds": round(m.avg_duration, 6),
            "EM": round(m.exact_match_rate, 3),
            "token_usage": self.agent.token_usage,
            "memory_entries": self.store.count(),
            "convergence_curve": self.optimizer.convergence_curve(),
        }
