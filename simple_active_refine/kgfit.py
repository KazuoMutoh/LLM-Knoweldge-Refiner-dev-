from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from simple_active_refine.util import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class KGFitEmbeddingPaths:
    """Paths to precomputed KG-FIT text embeddings and metadata."""

    name_embeddings: Path
    desc_embeddings: Path
    meta: Path


@dataclass(frozen=True)
class KGFitEmbeddingConfig:
    """Configuration for loading KG-FIT text embeddings."""

    paths: KGFitEmbeddingPaths
    reshape_strategy: str = "full"  # full | slice | project
    embedding_dim: Optional[int] = None  # used for slice/project


class KGFitEmbeddingError(RuntimeError):
    """Raised when KG-FIT embeddings or metadata are invalid."""


def _load_numpy_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise KGFitEmbeddingError(f"Embedding file not found: {path}")
    array = np.load(path)
    if not isinstance(array, np.ndarray):
        raise KGFitEmbeddingError(f"Embedding file is not a numpy array: {path}")
    if array.ndim != 2:
        raise KGFitEmbeddingError(f"Expected 2D embedding array, got shape={array.shape}")
    return array


def _load_entity_to_row(meta_path: Path) -> Dict[str, int]:
    if not meta_path.exists():
        raise KGFitEmbeddingError(f"Embedding meta file not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    entity_to_row = meta.get("entity_to_row")
    if not isinstance(entity_to_row, dict):
        raise KGFitEmbeddingError("meta['entity_to_row'] must be a dict")
    return {str(k): int(v) for k, v in entity_to_row.items()}


def load_kgfit_raw_embeddings(
    *,
    paths: KGFitEmbeddingPaths,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Load raw name/desc embeddings and ordered entity ids.

    Returns:
        (name_embeddings, desc_embeddings, entity_ids_in_row_order)
    """

    name_embeddings = _load_numpy_array(paths.name_embeddings)
    desc_embeddings = _load_numpy_array(paths.desc_embeddings)
    entity_to_row = _load_entity_to_row(paths.meta)

    num_rows = name_embeddings.shape[0]
    entity_ids = [None] * num_rows
    for entity, row_idx in entity_to_row.items():
        if row_idx < 0 or row_idx >= num_rows:
            raise KGFitEmbeddingError(
                f"entity_to_row has out-of-range index: {entity} -> {row_idx}"
            )
        entity_ids[row_idx] = entity

    if any(e is None for e in entity_ids):
        missing = sum(1 for e in entity_ids if e is None)
        raise KGFitEmbeddingError(
            f"entity_to_row missing {missing} rows for embedding size={num_rows}"
        )

    return name_embeddings, desc_embeddings, tuple(entity_ids)


def _reshape_embeddings(
    *,
    name_embeddings: np.ndarray,
    desc_embeddings: np.ndarray,
    reshape_strategy: str,
    embedding_dim: Optional[int],
) -> np.ndarray:
    if name_embeddings.shape != desc_embeddings.shape:
        raise KGFitEmbeddingError(
            "name/desc embedding shape mismatch: "
            f"{name_embeddings.shape} vs {desc_embeddings.shape}"
        )

    if reshape_strategy == "full":
        return np.concatenate([name_embeddings, desc_embeddings], axis=1)

    if reshape_strategy == "slice":
        if embedding_dim is None:
            raise KGFitEmbeddingError("embedding_dim is required for slice strategy")
        if embedding_dim % 2 != 0:
            raise KGFitEmbeddingError("embedding_dim must be even for slice strategy")
        half = embedding_dim // 2
        return np.concatenate([name_embeddings[:, :half], desc_embeddings[:, :half]], axis=1)

    if reshape_strategy == "project":
        raise KGFitEmbeddingError(
            "project strategy requires a trainable projection layer and is not supported "
            "in the loader. Use 'full' or 'slice' for initialization."
        )

    raise KGFitEmbeddingError(f"Unknown reshape_strategy: {reshape_strategy}")


def load_kgfit_entity_embeddings(
    *,
    entity_to_id: Dict[str, int],
    config: KGFitEmbeddingConfig,
    dtype: torch.dtype = torch.float32,
) -> torch.FloatTensor:
    """Load KG-FIT text embeddings aligned to the current entity mapping.

    Args:
        entity_to_id: Mapping from entity label to numeric id.
        config: KG-FIT embedding config with paths and reshape strategy.
        dtype: Torch dtype for returned tensor.

    Returns:
        Tensor of shape (num_entities, embedding_dim).
    """

    name_embeddings = _load_numpy_array(config.paths.name_embeddings)
    desc_embeddings = _load_numpy_array(config.paths.desc_embeddings)
    entity_to_row = _load_entity_to_row(config.paths.meta)

    rows = []
    missing = []
    for entity, entity_id in sorted(entity_to_id.items(), key=lambda kv: kv[1]):
        row_idx = entity_to_row.get(entity)
        if row_idx is None:
            missing.append(entity)
            continue
        rows.append(row_idx)

    if missing:
        raise KGFitEmbeddingError(
            f"Missing embeddings for {len(missing)} entities. Example: {missing[:3]}"
        )

    aligned_name = name_embeddings[rows]
    aligned_desc = desc_embeddings[rows]
    merged = _reshape_embeddings(
        name_embeddings=aligned_name,
        desc_embeddings=aligned_desc,
        reshape_strategy=config.reshape_strategy,
        embedding_dim=config.embedding_dim,
    )

    tensor = torch.as_tensor(merged, dtype=dtype)
    logger.info(
        "Loaded KG-FIT embeddings: shape=%s strategy=%s", tensor.shape, config.reshape_strategy
    )
    return tensor


def resolve_kgfit_paths(*, dir_triples: Path, override: Optional[Dict[str, str]] = None) -> KGFitEmbeddingPaths:
    """Resolve default KG-FIT embedding paths relative to dataset dir.

    Args:
        dir_triples: Dataset directory (contains train/valid/test).
        override: Optional dict with explicit paths: name_embeddings, desc_embeddings, meta.

    Returns:
        KGFitEmbeddingPaths with concrete paths.
    """

    if override is None:
        override = {}

    base = dir_triples / ".cache" / "kgfit"
    name_path = Path(override.get("name_embeddings", base / "entity_name_embeddings.npy"))
    desc_path = Path(override.get("desc_embeddings", base / "entity_desc_embeddings.npy"))
    meta_path = Path(override.get("meta", base / "entity_embedding_meta.json"))

    return KGFitEmbeddingPaths(
        name_embeddings=name_path,
        desc_embeddings=desc_path,
        meta=meta_path,
    )
