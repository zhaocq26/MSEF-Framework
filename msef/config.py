"""MSEF global configuration.

Centralizes the main parameter settings, quality-gate dimension weights,
manual-approval risk configuration, and the high-risk operation list.
All values can be overridden via ``MSEF_*`` environment variables or
``load_config(overrides)`` for parameter-sensitivity experiments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Five-stage definition: S = {plan, code, test, review, deliver}
# ---------------------------------------------------------------------------
STAGE_ORDER: List[str] = ["plan", "code", "test", "review", "deliver"]

STAGE_LABELS: Dict[str, str] = {
    "plan": "Planning",
    "code": "Coding",
    "test": "Testing",
    "review": "Review",
    "deliver": "Delivery",
}


# ---------------------------------------------------------------------------
# Three-level memory system
# ---------------------------------------------------------------------------
@dataclass
class MemoryConfig:
    """Parameters of the three-level memory system."""

    # Semantic retrieval similarity threshold θ_sim (0.75)
    similarity_threshold: float = 0.75
    # Number of candidate entries retrieved k (5)
    top_k: int = 5
    # Composite priority weights: similarity / usage frequency / recency / level
    priority_weights: Dict[str, float] = field(
        default_factory=lambda: {"sim": 0.4, "freq": 0.2, "recency": 0.2, "level": 0.2}
    )
    # Level weights: project / team / personal (1.0 / 0.6 / 0.3)
    level_weights: Dict[str, float] = field(
        default_factory=lambda: {"project": 1.0, "team": 0.6, "personal": 0.3}
    )
    # Recency time window ΔT
    recency_window_seconds: float = 30 * 24 * 3600.0
    # Activity score weights: frequency / recency / consistency
    activity_weights: Dict[str, float] = field(
        default_factory=lambda: {"freq": 0.4, "recency": 0.3, "consistency": 0.3}
    )
    # Lifecycle thresholds (downgrade/archive/eliminate): downgrade below median, archive below lower quartile
    downgrade_ratio: float = 0.5
    archive_ratio: float = 0.25
    eliminate_ratio: float = 0.05
    # Semantic embedding dimension (Sentence-BERT all-MiniLM-L6-v2 → 384)
    embedding_dim: int = 384
    # Similarity threshold for merge updates
    merge_similarity_threshold: float = 0.85

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryConfig":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dc_fields(cls)}})


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
# Per-stage quality gate thresholds τ_k (0.7 / 0.8 / 0.85 / 0.8 / 0.75)
DEFAULT_GATE_THRESHOLDS: Dict[str, float] = {
    "plan": 0.7,
    "code": 0.8,
    "test": 0.85,
    "review": 0.8,
    "deliver": 0.75,
}

# Per-stage evaluation dimensions and weights
DEFAULT_GATE_DIMENSIONS: Dict[str, Dict[str, float]] = {
    "plan": {
        "plan_completeness": 0.4,       # plan completeness
        "impact_coverage": 0.4,         # impact scope identification
        "risk_sufficiency": 0.2,        # risk assessment sufficiency
    },
    "code": {
        "syntax_correctness": 0.5,      # syntax correctness
        "compile_pass_rate": 0.3,       # compile pass rate
        "spec_compliance": 0.2,         # spec compliance
    },
    "test": {
        "unit_pass_rate": 0.4,          # unit test pass rate
        "integration_pass_rate": 0.3,   # integration test pass rate
        "regression_result": 0.3,       # regression test result (1 - new defect introduction rate)
    },
    "review": {
        "spec_compliance": 0.3,         # spec compliance
        "risk_score": 0.4,              # potential risk score
        "static_alert_count": 0.3,      # static analysis alert count
    },
    "deliver": {
        "report_completeness": 0.4,     # change report completeness
        "risk_marking_sufficiency": 0.4,  # risk item marking sufficiency
        "deliverable_completeness": 0.2,  # deliverable completeness
    },
}


# ---------------------------------------------------------------------------
# Manual approval
# ---------------------------------------------------------------------------
@dataclass
class ApprovalConfig:
    """Manual approval configuration."""

    # Risk threshold triggering manual approval τ_r (0.6)
    risk_threshold: float = 0.6
    # Stages where stage-level approval is mandatory; others are configurable
    stage_approval_stages: List[str] = field(
        default_factory=lambda: ["review", "deliver"]
    )
    # Stage-level risk assessment weights (change scale / critical module touch / high-risk item ratio)
    stage_risk_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    # Operation-level risk assessment weights (operation type / impact scope / reversibility)
    op_risk_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    # Approval timeout (seconds)
    approval_timeout_seconds: int = 24 * 3600
    # Whether approval records are persisted into the memory system
    record_to_memory: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ApprovalConfig":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dc_fields(cls)}})


# Risk level per operation type (extreme/high/medium/low/minimal) → default values
OP_RISK_LEVELS: Dict[str, Dict[str, Any]] = {
    "extreme": {"weight": 0.95, "label": "extreme risk"},
    "high": {"weight": 0.8, "label": "high risk"},
    "medium": {"weight": 0.6, "label": "medium risk"},
    "low": {"weight": 0.4, "label": "low risk"},
    "minimal": {"weight": 0.2, "label": "minimal risk"},
}

# Typical high-risk operations per stage with their risk levels
HIGH_RISK_OPERATIONS: Dict[str, Dict[str, Any]] = {
    # Planning stage
    "plan_pick_tech_solution": {"stage": "plan", "level": "extreme", "trigger": "architecture or technology-stack change"},
    "plan_define_impact": {"stage": "plan", "level": "medium", "trigger": "multiple modules or cross-service calls"},
    "plan_rollback_scheme": {"stage": "plan", "level": "high", "trigger": "irreversible operation or data migration"},
    # Coding stage
    "code_modify_config": {"stage": "code", "level": "extreme", "trigger": "production configuration or critical parameter"},
    "code_schema_change": {"stage": "code", "level": "extreme", "trigger": "schema change or data migration"},
    "code_dep_version": {"stage": "code", "level": "high", "trigger": "impact on downstream modules"},
    "code_auth_logic": {"stage": "code", "level": "extreme", "trigger": "security boundary change"},
    "code_api_signature": {"stage": "code", "level": "high", "trigger": "external callers exist"},
    "code_delete_file": {"stage": "code", "level": "medium", "trigger": "non-temporary file"},
    # Testing stage
    "test_modify_case": {"stage": "test", "level": "medium", "trigger": "core or security-related test case"},
    "test_skip_case": {"stage": "test", "level": "high", "trigger": "core or security-related test case"},
    "test_adjust_threshold": {"stage": "test", "level": "medium", "trigger": "lowered pass threshold"},
    "test_db_migration": {"stage": "test", "level": "extreme", "trigger": "irreversible operation"},
    # Review stage
    "review_waive_vuln": {"stage": "review", "level": "extreme", "trigger": "high-severity vulnerability or known risk"},
    "review_downgrade_risk": {"stage": "review", "level": "high", "trigger": "downgrade of a high-risk item"},
    "review_approve_exception": {"stage": "review", "level": "medium", "trigger": "coding-standard exception approved by a reviewer"},
    # Delivery stage
    "deliver_merge_trunk": {"stage": "deliver", "level": "high", "trigger": "all changes"},
    "deliver_deploy_prod": {"stage": "deliver", "level": "extreme", "trigger": "all changes"},
    "deliver_rollback": {"stage": "deliver", "level": "extreme", "trigger": "production rollback"},
}


# ---------------------------------------------------------------------------
# Model adaptation layer
# ---------------------------------------------------------------------------
@dataclass
class AdaptationConfig:
    """Model adaptation layer configuration."""

    # Context pre-compression target length (32k tokens)
    context_target_tokens: int = 32 * 1024
    # Model call temperature (0.2)
    temperature: float = 0.2
    # Decomposition granularity candidates (file → module+file → function+file → dynamic adaptive)
    granularity_levels: List[str] = field(
        default_factory=lambda: ["file", "module_file", "function_file", "adaptive"]
    )
    # Candidate model pool
    models: List[str] = field(
        default_factory=lambda: [
            "deepseek-v4-flash",   # local model (Ascend compute cluster)
            "qwen2.5-coder-32b",   # local model
            "gpt-4o",              # remote model (enterprise API gateway)
            "claude-3.5-sonnet",   # remote model
        ]
    )
    # Model scoring matrix: task type/stage → expected score per model (learnable; initialization given here)
    model_scores: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "plan": {"deepseek-v4-flash": 0.6, "qwen2.5-coder-32b": 0.65, "gpt-4o": 0.85, "claude-3.5-sonnet": 0.9},
            "code": {"deepseek-v4-flash": 0.7, "qwen2.5-coder-32b": 0.9, "gpt-4o": 0.85, "claude-3.5-sonnet": 0.8},
            "test": {"deepseek-v4-flash": 0.75, "qwen2.5-coder-32b": 0.8, "gpt-4o": 0.85, "claude-3.5-sonnet": 0.85},
            "review": {"deepseek-v4-flash": 0.55, "qwen2.5-coder-32b": 0.6, "gpt-4o": 0.9, "claude-3.5-sonnet": 0.92},
            "deliver": {"deepseek-v4-flash": 0.5, "qwen2.5-coder-32b": 0.55, "gpt-4o": 0.85, "claude-3.5-sonnet": 0.88},
        }
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdaptationConfig":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dc_fields(cls)}})


# ---------------------------------------------------------------------------
# Collaborative optimization mechanism
# ---------------------------------------------------------------------------
@dataclass
class OptimizationConfig:
    """Collaborative optimization mechanism parameters."""

    # Decomposition strategy learning rate η (0.01)
    learning_rate: float = 0.01
    # Strategy update frequency (every 50 tasks)
    update_every_tasks: int = 50
    # Hidden dimension of the experience loss prediction function
    hidden_dim: int = 32
    # Optimization objective weights: min w_t*time + w_i*intervention - w_q*quality
    weight_time: float = 0.1
    weight_intervention: float = 1.0
    weight_quality: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizationConfig":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dc_fields(cls)}})


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """MSEF global configuration (full parameterization)."""

    # Maximum retries on failure rollback
    max_retries: int = 3
    # Shared risk threshold for stage-level and operation-level approval
    risk_threshold: float = 0.6
    # Whether checkpoint resume is enabled
    checkpoint_enabled: bool = True
    # Whether failure rollbacks and approval records are written to the memory system
    audit_to_memory: bool = True
    # Sub-module configurations
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    approval: ApprovalConfig = field(default_factory=ApprovalConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    # Quality gate thresholds and dimension weights (overridable)
    gate_thresholds: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_GATE_THRESHOLDS)
    )
    gate_dimensions: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_GATE_DIMENSIONS.items()}
    )
    # Seed (reproducible decomposition strategy initialization and simulation)
    seed: int = 42

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        known = {f.name for f in dc_fields(cls)}
        cfg = cls()
        for k, v in d.items():
            if k not in known:
                continue
            if k in ("memory", "approval", "adaptation", "optimization") and isinstance(v, dict):
                setattr(cfg, k, type(getattr(cfg, k)).from_dict(v))
            elif k in ("gate_thresholds", "gate_dimensions") and isinstance(v, dict):
                setattr(cfg, k, v)
            else:
                setattr(cfg, k, v)
        return cfg


def _env_overrides() -> Dict[str, Any]:
    """Read overrides from ``MSEF_*`` environment variables (for scripted parameter scans)."""
    out: Dict[str, Any] = {}
    mapping = {
        "MSEF_RISK_THRESHOLD": ("risk_threshold", float),
        "MSEF_MAX_RETRIES": ("max_retries", int),
        "MSEF_SEED": ("seed", int),
        "MSEF_SIM_THRESHOLD": ("memory.similarity_threshold", float),
        "MSEF_TOP_K": ("memory.top_k", int),
        "MSEF_LEARNING_RATE": ("optimization.learning_rate", float),
    }
    for env, (path, cast) in mapping.items():
        val = os.environ.get(env)
        if val is not None:
            keys = path.split(".")
            cur = out
            for k in keys[:-1]:
                cur = cur.setdefault(k, {})
            cur[keys[-1]] = cast(val)
    return out


def load_config(overrides: Optional[Dict[str, Any]] = None, env: bool = True) -> Config:
    """Load configuration.

    Args:
        overrides: optional key-value overrides (dot paths supported, e.g. ``{"memory.top_k": 10}``).
        env: whether to merge ``MSEF_*`` environment variable overrides.
    """
    merged: Dict[str, Any] = {}
    if env:
        merged.update(_env_overrides())
    if overrides:
        for k, v in overrides.items():
            keys = k.split(".")
            cur = merged
            for kk in keys[:-1]:
                cur = cur.setdefault(kk, {})
            cur[keys[-1]] = v
    cfg = Config.from_dict(merged)
    cfg.gate_thresholds = dict(DEFAULT_GATE_THRESHOLDS)
    cfg.gate_dimensions = {k: dict(v) for k, v in DEFAULT_GATE_DIMENSIONS.items()}
    # Re-apply overrides
    if "gate_thresholds" in merged:
        cfg.gate_thresholds.update(merged["gate_thresholds"])
    if "gate_dimensions" in merged:
        cfg.gate_dimensions = merged["gate_dimensions"]
    return cfg
