"""MSEF: An LLM Agent-Driven Multi-Stage Execution Framework for Enterprise Software Engineering.

Core business logic is built with Python 3.9 and PyTorch, comprising four components:

1. Model adaptation layer (msef.adaptation): task reduction, context pre-compression, multi-model routing
2. Multi-stage execution engine (msef.engine): five stages of plan-code-test-review-deliver with embedded quality gates and human approval
3. Three-tier memory system (msef.memory): project-, team-, and personal-level memory storage, retrieval, and lifecycle management
4. Collaborative optimization mechanism (msef.optimization): three-way feedback loop across engine/memory/adaptation layers

Core business logic (reduction strategy parameters, experience loss, memory semantic vectors, dependency matrix operations, etc.) is implemented with PyTorch tensors.
"""

__version__ = "0.1.0"

from msef.config import load_config  # noqa: E402
from msef.types import (  # noqa: E402
    ApprovalMode,
    ApprovalVerdict,
    MemoryScope,
    OperationRiskLevel,
    Stage,
    Task,
)

__all__ = [
    "load_config",
    "ApprovalMode",
    "ApprovalVerdict",
    "MemoryScope",
    "OperationRiskLevel",
    "Stage",
    "Task",
    "__version__",
]