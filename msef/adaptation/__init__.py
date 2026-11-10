"""Model adaptation layer.

Task reduction φ(T; θ_d), context pre-compression ψ(C), and multi-model routing ρ convert complex tasks and raw context into model-executable input, serving as the input preprocessing layer of the execution engine. The reduction strategy parameters θ_d are PyTorch learnable parameters, updated by gradient descent from historical experience in the memory system via the collaborative optimization mechanism.
"""
from msef.adaptation.compression import ContextCompressor
from msef.adaptation.decomposition import TaskReducer, DecompositionPolicy
from msef.adaptation.router import ModelRouter

__all__ = [
    "ContextCompressor",
    "TaskReducer",
    "DecompositionPolicy",
    "ModelRouter",
]