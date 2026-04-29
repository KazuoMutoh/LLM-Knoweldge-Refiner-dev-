from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Pooling = Literal["cls", "mean", "max"]


@dataclass
class SimKGCConfig:
    """Configuration for SimKGC training/inference.

    This config is intentionally minimal and focuses on the knobs used in this
    repo's experiments. Defaults are chosen to be safe for CPU smoke tests.

    Notes:
        - `pretrained_model` can be a HuggingFace model name or a local path.
        - For unit tests (no external downloads), set `pretrained_model` to
          "__dummy__" to use a tiny hash-based encoder.
    """

    pretrained_model: str = "bert-base-uncased"
    pooling: Pooling = "mean"
    max_num_tokens: int = 50

    # Training
    train_batch_size: int = 64
    num_epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    device: Optional[str] = None

    # SimKGC-specific
    additive_margin: float = 0.02
    temperature: float = 0.05
    learnable_temperature: bool = True

    use_self_negative: bool = False
    self_negative_weight: float = 1.0

    # Evaluation
    eval_batch_size: int = 256

    # For offline tests
    embedding_dim: int = 128

    # Internal
    seed: Optional[int] = 42

    def to_dict(self) -> dict:
        return {
            "pretrained_model": self.pretrained_model,
            "pooling": self.pooling,
            "max_num_tokens": self.max_num_tokens,
            "train_batch_size": self.train_batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "additive_margin": self.additive_margin,
            "temperature": self.temperature,
            "learnable_temperature": self.learnable_temperature,
            "use_self_negative": self.use_self_negative,
            "self_negative_weight": self.self_negative_weight,
            "eval_batch_size": self.eval_batch_size,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SimKGCConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        # Backward-compatible key mapping
        remap = {
            "batch_size": "train_batch_size",
            "epochs": "num_epochs",
            "lr": "learning_rate",
            "t_init": "temperature",
            "finetune_t": "learnable_temperature",
        }
        normalized = {remap.get(k, k): v for k, v in payload.items()}
        filtered = {k: v for k, v in normalized.items() if k in allowed}
        return cls(**filtered)
