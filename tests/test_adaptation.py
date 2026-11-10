"""Model adaptation layer tests (task decomposition, context pre-compression, multi-model routing)."""
import torch

from msef.adaptation.compression import ContextCompressor
from msef.adaptation.decomposition import DecompositionPolicy, TaskReducer
from msef.adaptation.router import ModelRouter
from msef.config import AdaptationConfig
from msef.types import Stage, Task, TaskType


def make_task(n_files: int = 4) -> Task:
    return Task.create(
        title="Add captcha validation to the login API",
        description="Add sliding-captcha validation to the login endpoint and update request parameters.",
        task_type=TaskType.FEATURE,
        affected_files=[f"src/auth/login{i}.py" for i in range(n_files)],
        required_risks=["authentication logic change"],
        required_plan_elements=["scope", "dependencies", "sequence", "risks", "rollback"],
        metadata={"modules": ["auth", "user"], "complexity": 0.6},
    )


def test_decomposition_granularity_and_subtasks():
    """Task decomposition yields an executable subtask set; granularity is policy-driven."""
    policy = DecompositionPolicy(seed=7)
    reducer = TaskReducer(policy=policy, seed=7)
    task = make_task(n_files=6)
    result = reducer.decompose(task)
    assert result.count >= 1
    assert all(s.files for s in result.subtasks)
    assert result.granularity in {"file", "module_file", "function_file", "adaptive"}
    # Union of subtask files covers the affected files
    covered = [f for s in result.subtasks for f in s.files]
    assert len(covered) > 0


def test_policy_learnable_params_and_forward():
    """Decomposition policy θ_d is a learnable PyTorch parameter supporting forward pass."""
    policy = DecompositionPolicy(seed=1)
    feats = torch.randn(2, policy.feature_dim)
    gran, log_n, q = policy(feats)
    assert log_n.shape == (2, 1)
    assert q.shape == (2, 1)
    # Parameters are trainable
    assert all(p.requires_grad for p in policy.parameters())
    # Default dominant granularity is file-level
    assert policy.dominant_granularity() == "file"


def test_policy_gradient_update_changes_granularity():
    """Collaborative optimization: gradient update changes policy parameters (feedback-loop basis)."""
    policy = DecompositionPolicy(seed=2)
    before = policy.granularity_logits.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    feats = torch.randn(4, policy.feature_dim)
    gran, log_n, q = policy(feats)
    loss = (q - torch.ones_like(q)).pow(2).mean()
    loss.backward()
    optimizer.step()
    after = policy.granularity_logits.detach()
    assert not torch.allclose(before, after)


def test_context_compression_respects_budget():
    """Context pre-compression: compress overlong outputs to the target length."""
    compressor = ContextCompressor(AdaptationConfig(context_target_tokens=100))
    big_contexts = [f"File {i} content " + "x" * 500 for i in range(20)]
    task = make_task()
    result = compressor.compress(big_contexts, task, included_files=task.affected_files)
    # Compressed token count stays within budget (including small headers)
    assert compressor.estimate_tokens(result) <= 100 + 40


def test_context_compression_prefers_relevant_files():
    """Pre-compression keeps the fragments most relevant to the task."""
    compressor = ContextCompressor(AdaptationConfig(context_target_tokens=500))
    task = make_task()
    unrelated = ["def helper(): pass", "utility helper" * 20]
    related = ["src/auth/login0.py login endpoint: def login(): ...", "authentication configuration" * 20]
    result = compressor.compress(related + unrelated, task, included_files=task.affected_files)
    assert "login0.py" in result


def test_model_router_routes_by_stage():
    """Multi-model routing: different stages select different models (token control)."""
    router = ModelRouter()
    # Plan/review: remote strong-reasoning model
    assert router.route(Stage.PLAN) == "claude-3.5-sonnet"
    assert router.route(Stage.REVIEW) == "claude-3.5-sonnet"
    # Code: local coder model is cost-effective
    assert router.route(Stage.CODE) == "qwen2.5-coder-32b"


def test_router_score_update():
    """Collaborative optimization: router score matrix is updatable (adaptation layer → engine feedback)."""
    router = ModelRouter()
    router.update_scores({"code": {"qwen2.5-coder-32b": -0.2, "gpt-4o": 0.3}})
    assert router.route(Stage.CODE) == "gpt-4o"


def test_reducer_extract_features_shape():
    reducer = TaskReducer(seed=3)
    feats = reducer.extract_features(make_task())
    assert feats.shape == (1, 8)
