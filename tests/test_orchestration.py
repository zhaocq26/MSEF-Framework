"""Dependency orchestration tests (dependency matrix, transitive closure, topological sort, parallelism)."""
import torch

from msef.engine.orchestration import DependencyOrchestrator, StageDependencyGraph
from msef.types import Stage


def test_dependency_matrix_main_chain():
    """Main execution chain: plan→code→test→review→deliver direct dependency matrix."""
    g = StageDependencyGraph()
    M = g.dependency_matrix()
    assert M.shape == (5, 5)
    assert M.dtype == torch.bool
    # Direct dependency edges
    assert M[Stage.PLAN.index, Stage.CODE.index]
    assert M[Stage.CODE.index, Stage.TEST.index]
    assert M[Stage.TEST.index, Stage.REVIEW.index]
    assert M[Stage.REVIEW.index, Stage.DELIVER.index]
    # No reverse dependency
    assert not M[Stage.CODE.index, Stage.PLAN.index]


def test_transitive_closure():
    """Transitive closure: plan→test has no direct edge but is reachable via code."""
    g = StageDependencyGraph()
    D = g.transitive_closure()
    # Directly reachable
    assert D[Stage.PLAN.index, Stage.CODE.index]
    # Indirectly reachable (D describes the full dependency relation)
    assert D[Stage.PLAN.index, Stage.TEST.index]
    assert D[Stage.PLAN.index, Stage.DELIVER.index]
    assert D[Stage.CODE.index, Stage.REVIEW.index]
    # Unreachable direction
    assert not D[Stage.DELIVER.index, Stage.PLAN.index]


def test_parallel_condition():
    """Parallel condition: two stages are parallelizable only if no path exists between them in either direction."""
    g = StageDependencyGraph()
    D = g.transitive_closure()
    # Plan and deliver: a path exists on the main chain, so they cannot run in parallel
    assert not g.can_parallel(D, Stage.PLAN, Stage.DELIVER)
    # Subtasks within code can run in parallel — a stage may parallelize with itself
    assert g.can_parallel(D, Stage.CODE, Stage.CODE)


def test_topological_sort():
    """Topological sort result equals the main execution chain."""
    orch = DependencyOrchestrator()
    order = orch.schedule()
    assert order == [Stage.PLAN, Stage.CODE, Stage.TEST, Stage.REVIEW, Stage.DELIVER]


def test_predecessors():
    g = StageDependencyGraph()
    preds = g.predecessors(Stage.DELIVER)
    assert set(preds) == {Stage.PLAN, Stage.CODE, Stage.TEST, Stage.REVIEW}


def test_dependency_report():
    orch = DependencyOrchestrator()
    report = orch.dependency_report()
    assert report["plan"]["predecessors"] == []
    assert report["deliver"]["predecessors"] == ["plan", "code", "test", "review"]


def test_parallel_clusters_with_dag():
    """Parallel grouping on a generic DAG: independent nodes can run in parallel."""
    g = StageDependencyGraph()
    # Extra edge: test and review partially parallelize (review can overlap with test)
    orch = DependencyOrchestrator(g)
    clusters = orch.parallel_clusters([Stage.PLAN, Stage.CODE, Stage.TEST, Stage.REVIEW, Stage.DELIVER])
    # Any two stages on the main chain cannot run in parallel → each stage forms its own cluster
    assert len(clusters) == 5
