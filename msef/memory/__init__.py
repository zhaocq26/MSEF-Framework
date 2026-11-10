"""Three-tier memory system.

Knowledge is organized into project-level (M_p), team-level (M_t), and
personal-level (M_l) scopes, with semantic retrieval plus update and lifecycle
management. Cosine-similarity search is implemented with PyTorch; the default
semantic encoder is Sentence-BERT (all-MiniLM-L6-v2, 384-dim), falling back to
a deterministic feature-based encoder when unavailable for offline use.
"""
from msef.memory.encoder import SemanticEncoder
from msef.memory.entry import MemoryEntry
from msef.memory.lifecycle import MemoryLifecycleManager, MemoryLifecycle
from msef.memory.retrieval import MemoryRetriever
from msef.memory.storage import MemoryStore

__all__ = [
    "SemanticEncoder",
    "MemoryEntry",
    "MemoryLifecycleManager",
    "MemoryLifecycle",
    "MemoryRetriever",
    "MemoryStore",
]