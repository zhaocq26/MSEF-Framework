"""Memory entry.

A memory entry is the basic unit of the three-tier memory system, modeled as
a triple (content, scope, vector) with metadata (creation time, last access
time, usage count, source type, consistency score, etc.) used for retrieval
prioritization and lifecycle management. The definition lives in
:mod:`msef.types`.
"""
from msef.types import MemoryEntry

__all__ = ["MemoryEntry"]