"""Dependency orchestration and parallel execution.

The five stages form a directed acyclic graph G=(S, E) with S stage nodes and
E dependency edges. In the dependency matrix M, M[i][j]=1 means stage i must
run before stage j; the transitive closure M⁺ is computed by accumulating
matrix powers (D[i][j]=1 means i reaches j via some path).

Matrix powers are implemented with PyTorch boolean tensors (logical or/and),
giving the parallel condition:

    i, j can run in parallel ⇔ D[i][j]=0 and D[j][i]=0
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from msef.types import Stage


class StageDependencyGraph:
    """Five-stage dependency graph and dependency matrix.

    Main chain: plan→code→test→review→deliver. Subtasks within code can run in
    parallel, test suites within test can run in parallel, and review can
    partially overlap with test.
    """

    def __init__(self, stages: Optional[List[Stage]] = None) -> None:
        self.stages = stages or Stage.ordered()
        self.n = len(self.stages)
        self.index = {s: i for i, s in enumerate(self.stages)}
        self._edges: List[Tuple[int, int]] = []
        self._default_edges()

    def _default_edges(self) -> None:
        """Default main-chain dependencies (plan←code←test←review←deliver)."""
        order = self.stages
        for i in range(len(order) - 1):
            self.add_edge(order[i], order[i + 1])

    def add_edge(self, pre: Stage, post: Stage) -> None:
        self._edges.append((self.index[pre], self.index[post]))

    # ------------------------------------------------------------------
    def dependency_matrix(self) -> torch.Tensor:
        """Direct dependency matrix M (bool tensor [n, n])."""
        m = torch.zeros((self.n, self.n), dtype=torch.bool)
        for i, j in self._edges:
            m[i, j] = True
        return m

    def transitive_closure(self) -> torch.Tensor:
        """Transitive closure D = M⁺ (accumulated matrix powers with logical or)."""
        m = self.dependency_matrix()
        # Warshall's algorithm (equivalent to boolean matrix power accumulation)
        d = m.clone()
        for k in range(self.n):
            for i in range(self.n):
                if d[i, k]:
                    d[i] = d[i] | d[k]
        return d

    def can_parallel(self, closure: torch.Tensor, a: Stage, b: Stage) -> bool:
        """Two stages can run in parallel ⇔ no reachable path in either direction."""
        i, j = self.index[a], self.index[b]
        return not (bool(closure[i, j]) or bool(closure[j, i]))

    def predecessors(self, stage: Stage) -> List[Stage]:
        idx = self.index[stage]
        closure = self.transitive_closure()
        preds = []
        for s, i in self.index.items():
            if closure[i, idx] and s != stage:
                preds.append(s)
        return preds


class DependencyOrchestrator:
    """Stage scheduler: topological sort + parallel condition detection.

    - Build the dependency graph and compute the dependency matrix
    - Topologically sort the DAG to get the stage execution sequence
    - Execute in sequence; stages (sets) satisfying the parallel condition may be scheduled in parallel
    - After each stage, check that all prerequisites of successor stages are satisfied
    """

    def __init__(self, graph: Optional[StageDependencyGraph] = None) -> None:
        self.graph = graph or StageDependencyGraph()
        self.closure = self.graph.transitive_closure()

    # ------------------------------------------------------------------
    def topological_order(self) -> List[Stage]:
        """Kahn topological sort."""
        m = self.graph.dependency_matrix()
        n = self.graph.n
        indeg = m.sum(dim=0).tolist()
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(n):
                if bool(m[i, j]):
                    adj[i].append(j)
        from collections import deque

        queue = deque([i for i in range(n) if indeg[i] == 0])
        order: List[int] = []
        while queue:
            i = queue.popleft()
            order.append(i)
            for j in adj[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    queue.append(j)
        if len(order) != n:  # pragma: no cover - DAG is guaranteed acyclic
            raise RuntimeError("Dependency graph contains a cycle; topological sort is impossible")
        stages = self.graph.stages
        return [stages[i] for i in order]

    # ------------------------------------------------------------------
    def parallel_clusters(self, stages: Sequence[Stage]) -> List[List[Stage]]:
        """Group a stage sequence by the parallel condition.

        Stages within a group are pairwise parallel; groups run serially in
        topological order. Simplified implementation: greedy merge in order.
        """
        clusters: List[List[Stage]] = []
        for s in stages:
            placed = False
            for cluster in clusters:
                if all(self.graph.can_parallel(self.closure, s, c) for c in cluster):
                    cluster.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s])
        return clusters

    def schedule(self) -> List[Stage]:
        """Return the topologically ordered stage sequence (main-chain plan)."""
        return self.topological_order()

    def dependency_report(self) -> Dict[str, object]:
        """Generate a structured dependency report."""
        order = self.schedule()
        out = {}
        for s in order:
            preds = self.graph.predecessors(s)
            peers = [p for p in order if p != s and self.graph.can_parallel(self.closure, s, p)]
            out[s.value] = {
                "predecessors": [p.value for p in preds],
                "parallel_stages": [p.value for p in peers],
            }
        return out
