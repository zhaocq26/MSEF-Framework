"""Task reduction.

Task reduction decomposes a complex task T into a set of model-executable subtasks:

    φ(T; θ_d) = {t_1, t_2, ..., t_n}

θ_d are learnable reduction strategy parameters controlling the granularity and manner of reduction (file-level → module+file-level → function+file-level → dynamic adaptive mixed reduction). θ_d is modeled as a PyTorch ``nn.Module`` (``DecompositionPolicy``): the forward pass outputs per-granularity logits, from which the current task's reduction granularity is sampled via softmax; θ_d also participates in predicting the experience loss ``L_mem``, supporting gradient updates by the collaborative optimization mechanism based on historical experience in the memory system.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from msef.config import AdaptationConfig
from msef.types import DecompositionResult, Subtask, Task

logger = logging.getLogger(__name__)

GRANULARITY_SCALE = {  # Split scale per granularity (maps θ_d heatmap to subtask count)
    "file": 1.0,
    "module_file": 0.75,
    "function_file": 0.5,
    "adaptive": 0.3,
}


class DecompositionPolicy(nn.Module):
    """Reduction strategy network (PyTorch core business logic).

    Holds learnable parameters θ_d:

    - ``granularity_logits``: preference per reduction granularity
    - ``split_head``: maps [task features + granularity representation] to a predicted subtask count (log scale)
    - ``quality_head``: part of the prediction function f(T, θ_d), estimating expected execution quality for a given task and strategy

    Forward returns::

        (granularity_logits, n_subtasks_pred, quality_pred)
    """

    def __init__(self, n_granularities: int = 4, hidden_dim: int = 32,
                 feature_dim: int = 8, init_granularity: Optional[str] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.n_granularities = n_granularities
        self.feature_dim = feature_dim
        gran_idx = {
            "file": 0, "module_file": 1, "function_file": 2, "adaptive": 3,
        }.get(init_granularity, 0)
        # Initialize with "file-level" preference (round 1: fixed file-level reduction)
        logits = torch.zeros(n_granularities)
        logits[gran_idx] = 2.0
        self.granularity_logits = nn.Parameter(logits.clone())
        self.split_head = nn.Sequential(
            nn.Linear(feature_dim + n_granularities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim + n_granularities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: Task features [B, feature_dim] (normalized task attributes).

        Returns:
            (granularity_logits, log_n_subtasks, quality_pred)
        """
        gran = self.granularity_logits.unsqueeze(0).expand(features.size(0), -1)
        fused = torch.cat([features, gran], dim=-1)
        log_n = self.split_head(fused)
        q = self.quality_head(fused)
        return self.granularity_logits, log_n, q

    def granularity_probs(self) -> torch.Tensor:
        return torch.softmax(self.granularity_logits, dim=-1)

    def dominant_granularity(self) -> str:
        names = ["file", "module_file", "function_file", "adaptive"]
        idx = int(torch.argmax(self.granularity_logits).item())
        return names[idx] if idx < len(names) else names[-1]

    def copy_theta(self) -> "DecompositionPolicy":
        """Deep-copy the current strategy parameters (for checkpoint resume / experiment snapshots)."""
        import copy

        cp = DecompositionPolicy(
            n_granularities=self.n_granularities,
            hidden_dim=self.split_head[0].out_features,
            feature_dim=self.feature_dim,
        )
        cp.load_state_dict(copy.deepcopy(self.state_dict()))
        return cp

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DecompositionPolicy dominant={self.dominant_granularity()} "
            f"probs={self.granularity_probs().detach().tolist()}>"
        )


class TaskReducer:
    """Task reducer: combines the strategy network with task splitting rules.

    Task reduction is controlled by θ_d in granularity and manner. Beyond the strategy network, an engineering-oriented subtask list is produced from the task's actual affected files, code directory structure, and task category, so the reduction result is both driven by the learnable policy and aligned with the physical boundaries of an enterprise codebase (subtasks are bounded by code files and module structure).
    """

    def __init__(self, policy: Optional[DecompositionPolicy] = None,
                 config: Optional[AdaptationConfig] = None, seed: Optional[int] = None) -> None:
        self.config = config or AdaptationConfig()
        self.policy = policy or DecompositionPolicy(
            n_granularities=len(self.config.granularity_levels),
            seed=seed,
        )
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def extract_features(self, task: Task) -> torch.Tensor:
        """Extract normalized task features (for the strategy network).

        Features include: task difficulty (number of affected files), task-type encoding, risk scale, etc.
        """
        n_files = float(len(task.affected_files)) if task.affected_files else 1.0
        n_risks = float(len(task.required_risks))
        type_map = {"feature": 0.0, "bugfix": 0.5, "refactor": 1.0}
        ttype = type_map.get(task.task_type.value, 0.0)
        n_criteria = float(len(task.acceptance_criteria))
        # Normalize: log(1+n) compresses the long tail
        feats = torch.tensor(
            [
                math.log1p(n_files) / 5.0,
                min(1.0, n_files / 20.0),
                n_risks / 10.0,
                ttype,
                min(1.0, n_criteria / 10.0),
                min(1.0, (len(task.metadata.get("modules", [])) or 1.0) / 10.0),
                float(task.metadata.get("complexity", 0.5)),
                float(task.metadata.get("history_failure_rate", 0.3)),
            ],
            dtype=torch.float32,
        )
        return feats.unsqueeze(0)

    def decompose(self, task: Task) -> DecompositionResult:
        """Run task reduction, returning the subtask set and a policy snapshot."""
        with torch.no_grad():
            features = self.extract_features(task)
            _, log_n, _ = self.policy(features)
        probs = self.policy.granularity_probs().detach()
        granularity = self.policy.dominant_granularity()

        # Determine subtask count from granularity scale and task size (avoid excessive calls / context fragmentation)
        base = GRANULARITY_SCALE.get(granularity, 1.0)
        n_est = max(1, int(round(len(task.affected_files) * base))) if task.affected_files else 1
        n_pred = max(1, int(torch.exp(log_n).item()))
        n_sub = max(1, min(n_est, n_pred, 12))
        if not task.affected_files:
            n_sub = 1

        subtasks = self._build_subtasks(task, n_sub, granularity)
        return DecompositionResult(subtasks=subtasks, granularity=granularity,
                                   theta=self.policy.copy_theta())

    def _build_subtasks(self, task: Task, n: int, granularity: str) -> List[Subtask]:
        """Build subtasks at file/module/function scale (simulated engineering split)."""
        files = list(task.affected_files or [f"{task.id}_file.py"])
        if n >= len(files):
            # One subtask per file
            return [
                Subtask.create(
                    description=f"Subtask {i}: process file {f} ({granularity} granularity)",
                    files=[f],
                    granularity=granularity,
                    risk_flags=[r for r in task.required_risks if i % 2 == 0][:1],
                )
                for i, f in enumerate(files[:n])
            ]
        # Split files into chunks
        per = math.ceil(len(files) / n)
        out = []
        for i in range(n):
            chunk = files[i * per : (i + 1) * per]
            out.append(Subtask.create(
                description=f"Subtask {i}: process file group {chunk} ({granularity} granularity)",
                files=chunk,
                granularity=granularity,
                risk_flags=task.required_risks[:1] if i == 0 else [],
            ))
        return out

    # -- Interface for the collaborative optimization mechanism ------------------
    def set_config_granularity(self, granularity: str) -> None:
        """Constrain the policy network's preferred granularity to a given value (for experiment alignment / ablation)."""
        names = ["file", "module_file", "function_file", "adaptive"]
        if granularity not in names:
            raise ValueError(f"Unknown granularity: {granularity}")
        with torch.no_grad():
            self.policy.granularity_logits.fill_(-1.0)
            self.policy.granularity_logits[names.index(granularity)] = 3.0
