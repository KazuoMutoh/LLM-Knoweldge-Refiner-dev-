"""SimKGC implementation.

This package provides a lightweight, repo-native implementation of the key
ideas in SimKGC (ACL 2022): bi-encoder scoring with InfoNCE training and large
negative sets (in-batch, optional pre-batch, optional self-negative).

The implementation is designed to integrate with `KnowledgeGraphEmbedding`
(via `embedding_backend="simkgc"`).
"""

from .config import SimKGCConfig
from .train import train_simkgc
from .wrapper import SimKGCWrapper

__all__ = [
    "SimKGCConfig",
    "SimKGCWrapper",
    "train_simkgc",
]
