"""Collaborative optimization tests (experience loss function, three-way feedback loop, convergence)."""
import torch

from msef.adaptation.decomposition import DecompositionPolicy, TaskReducer
from msef.config import OptimizationConfig
from msef.optimization.feedback import CollaborativeOptimizer, ExperienceLoss
from msef.types import Task, TaskType, TaskExecutionResult


def make_task(i: int = 0) -> Task:
    return Task.create(
        title=f"Task {i}: fix payment concurrency safety",
        description="Fix the idempotency defect in the payment callback.",
        task_type=TaskType.BUGFIX,
        affected_files=[f"src/pay/pay_{i}.py", "src/pay/common.py"],
        required_risks=["concurrency risk"],
        metadata={"complexity": 0.4 + 0.05 * i},
    )


def make_result(task: Task, score: float) -> TaskExecutionResult:
    r = TaskExecutionResult(task=task, completed=True, final_score=score,
                            duration_seconds=20.0)
    return r


def test_experience_loss_forward_backward():
    """Experience loss L_mem supports forward/backward (MSE scalar)."""
    policy = DecompositionPolicy(seed=0)
    loss_fn = ExperienceLoss(policy)
    feats = torch.randn(6, policy.feature_dim)
    actual = torch.tensor([0.6, 0.7, 0.8, 0.75, 0.9, 0.85])
    loss = loss_fn(feats, actual)
    loss.backward()
    assert loss.item() >= 0.0
    # All parameters receive gradients
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) >= 3


def test_sgd_reduces_experience_loss():
    """Gradient descent decreases experience loss over iterations (convergence)."""
    reducer = TaskReducer(policy=DecompositionPolicy(seed=0), seed=0)
    opt = CollaborativeOptimizer(
        reducer=reducer,
        config=OptimizationConfig(learning_rate=0.05, update_every_tasks=50),
    )
    for i in range(12):
        task = make_task(i)
        feats = reducer.extract_features(task).squeeze(0)
        opt.samples.append(_make_sample(task, feats, 0.5 + 0.04 * i))

    loss0 = opt.update_policy()
    loss1 = opt.update_policy()
    # Experience loss decreases after multiple SGD updates (monotonic convergence trend)
    assert loss1 < loss0
    assert len(opt.loss_history) == 2


def _make_sample(task, feats, score):
    from msef.optimization.feedback import ExperienceSample

    return ExperienceSample(task_id=task.id, features=feats, actual_score=score)


def test_collaborative_optimizer_collect_and_update(tmp_path):
    """Engine results flow into the optimizer; policy update triggers when the period is reached."""
    reducer = TaskReducer(policy=DecompositionPolicy(seed=1), seed=1)
    opt = CollaborativeOptimizer(
        reducer=reducer,
        config=OptimizationConfig(learning_rate=0.01, update_every_tasks=3),
    )
    for i in range(6):
        task = make_task(i)
        result = make_result(task, 0.6 + 0.05 * (i % 3))
        opt.collect(result)
    assert len(opt.samples) == 6
    updated = opt.maybe_update(completed_tasks=6)   # 6 % 3 == 0 → update
    assert updated
    assert opt.loss_history
    curve = opt.convergence_curve()
    assert curve["round"] >= 1
    assert len(curve["loss"]) == len(curve["em"]) == len(curve["granularity"])


def test_optimizer_skips_update_outside_period():
    reducer = TaskReducer(policy=DecompositionPolicy(seed=2), seed=2)
    opt = CollaborativeOptimizer(reducer=reducer,
                                 config=OptimizationConfig(update_every_tasks=5))
    for i in range(4):
        task = make_task(i)
        opt.collect(make_result(task, 0.7))
    assert not opt.maybe_update(completed_tasks=4)
    assert opt.loss_history == []


def test_policy_copy_independent():
    """Policy parameter snapshots are independent (checkpoint resume / experiment snapshots)."""
    policy = DecompositionPolicy(seed=3)
    cp = policy.copy_theta()
    with torch.no_grad():
        policy.granularity_logits += 0.5
    assert not torch.allclose(policy.granularity_logits, cp.granularity_logits)
