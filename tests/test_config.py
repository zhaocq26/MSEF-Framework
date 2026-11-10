"""Configuration tests: verify default values of the main parameters."""
import pytest

from msef.config import (
    ApprovalConfig,
    Config,
    DEFAULT_GATE_DIMENSIONS,
    DEFAULT_GATE_THRESHOLDS,
    load_config,
)


def test_default_gate_thresholds_match_reference_configuration():
    """Quality gate thresholds are 0.7 / 0.8 / 0.85 / 0.8 / 0.75."""
    assert DEFAULT_GATE_THRESHOLDS == {
        "plan": 0.7,
        "code": 0.8,
        "test": 0.85,
        "review": 0.8,
        "deliver": 0.75,
    }


def test_gate_dimension_weights_sum_to_one():
    """Per-stage evaluation dimension weights satisfy normalization Σ w = 1."""
    for stage, dims in DEFAULT_GATE_DIMENSIONS.items():
        assert abs(sum(dims.values()) - 1.0) < 1e-6, f"{stage} weights are not normalized: {dims}"


def test_priority_and_level_weights():
    cfg = load_config()
    pw = cfg.memory.priority_weights
    assert abs(sum(pw.values()) - 1.0) < 1e-6
    # Level weights: project/team/personal = 1.0/0.6/0.3
    assert cfg.memory.level_weights == {"project": 1.0, "team": 0.6, "personal": 0.3}
    assert cfg.memory.similarity_threshold == 0.75
    assert cfg.memory.top_k == 5


def test_optimization_defaults():
    cfg = load_config()
    assert cfg.optimization.learning_rate == 0.01
    assert cfg.optimization.update_every_tasks == 50
    assert cfg.risk_threshold == 0.6
    assert cfg.adaptation.context_target_tokens == 32 * 1024
    assert cfg.adaptation.temperature == 0.2


def test_approval_default_config():
    cfg = load_config()
    assert isinstance(cfg.approval, ApprovalConfig)
    assert cfg.approval.risk_threshold == 0.6
    # Default stage-level approval is mandatory at review and deliver
    assert set(cfg.approval.stage_approval_stages) == {"review", "deliver"}


def test_load_config_overrides_and_env(monkeypatch):
    monkeypatch.setenv("MSEF_TOP_K", "10")
    monkeypatch.setenv("MSEF_RISK_THRESHOLD", "0.5")
    cfg = load_config({"gate_thresholds": {"code": 0.9}, "memory.top_k": 7})
    # Explicit overrides take precedence over environment variables
    assert cfg.memory.top_k == 7
    assert cfg.risk_threshold == 0.5
    assert cfg.gate_thresholds["code"] == 0.9  # explicit override
    assert cfg.gate_thresholds["plan"] == 0.7  # others keep defaults

    # Read when only environment variables apply
    cfg2 = load_config({})
    assert cfg2.memory.top_k == 10
    assert cfg2.risk_threshold == 0.5
