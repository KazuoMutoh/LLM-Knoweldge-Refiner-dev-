from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from simple_active_refine.kgfit_precompute import (
    KGFitPrecomputeConfig,
    _ensure_complete_texts,
    build_embedder,
    embed_texts,
    read_entities_file,
    read_entity_texts,
    resolve_entity_order,
)
from simple_active_refine.io_utils import read_triples
from simple_active_refine.util import get_logger

logger = get_logger(__name__)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _reshape_merged_embeddings(
    *,
    name_emb: np.ndarray,
    desc_emb: np.ndarray,
    reshape_strategy: str,
    embedding_dim: Optional[int],
) -> np.ndarray:
    if reshape_strategy == "full":
        return np.concatenate([name_emb, desc_emb], axis=1)
    if reshape_strategy == "slice":
        if embedding_dim is None or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for slice strategy")
        half = embedding_dim // 2
        return np.concatenate([name_emb[:, :half], desc_emb[:, :half]], axis=1)
    raise ValueError(f"Unknown reshape_strategy: {reshape_strategy}")


def _cosine_sim_to_centers(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32, copy=False)
    centers = centers.astype(np.float32, copy=False)
    v_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    c_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    return v_norm @ c_norm.T


def ensure_kgfit_cache_complete(
    *,
    dir_triples: Path,
    cache_dir: Optional[Path] = None,
    reshape_strategy: str = "full",
    embedding_dim: Optional[int] = None,
    precompute_config: Optional[KGFitPrecomputeConfig] = None,
    name_source: str = "entity2text.txt",
    desc_source: str = "entity2textlong.txt",
) -> None:
    """Ensure KG-FIT cache covers all entities in a dataset.

    This is a lightweight *incremental* updater for KG-FIT artifacts:
      - Extends entity name/desc embeddings + meta for any missing entities.
      - Extends seed hierarchy labels for any missing entities by assigning them
        to the nearest existing cluster center (cosine similarity).

    Notes:
      - This intentionally does *not* recompute cluster centers or neighbor
        clusters (approximation), to avoid O(n^2) clustering.
      - Requires entity texts in `entity2text*.txt`.

    Args:
        dir_triples: Dataset directory containing train/valid/test + entity texts.
        cache_dir: KG-FIT cache directory (default: <dir_triples>/.cache/kgfit).
        reshape_strategy: "full" or "slice" (must match KG-FIT config).
        embedding_dim: Required when reshape_strategy == "slice".
        precompute_config: Controls embedding model/batch size.
        name_source: TSV filename under dir_triples for entity name text.
        desc_source: TSV filename under dir_triples for entity long text.
    """

    cache_dir = cache_dir or (dir_triples / ".cache" / "kgfit")
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "entity_embedding_meta.json"
    name_path = cache_dir / "entity_name_embeddings.npy"
    desc_path = cache_dir / "entity_desc_embeddings.npy"

    if not (meta_path.exists() and name_path.exists() and desc_path.exists()):
        raise FileNotFoundError(
            "KG-FIT cache is incomplete. Expected: "
            f"{meta_path}, {name_path}, {desc_path}"
        )

    meta = _load_json(meta_path)
    entity_to_row = meta.get("entity_to_row")
    if not isinstance(entity_to_row, dict):
        raise ValueError("meta['entity_to_row'] must be a dict")
    entity_to_row = {str(k): int(v) for k, v in entity_to_row.items()}

    name_emb = np.load(name_path)
    desc_emb = np.load(desc_path)
    if name_emb.shape != desc_emb.shape:
        raise ValueError(
            f"KG-FIT name/desc embedding shape mismatch: {name_emb.shape} vs {desc_emb.shape}"
        )

    name_texts = read_entity_texts(dir_triples / name_source)
    desc_texts = read_entity_texts(dir_triples / desc_source)

    # Determine required entity set as robustly as possible.
    # - Prefer entities.txt when present
    # - Fall back to entity2text sources
    # - Always include entities observed in train/valid/test triples
    entities_from_file = read_entities_file(dir_triples / "entities.txt")
    entities: List[str]
    if entities_from_file:
        entities = list(entities_from_file)
    else:
        entities = resolve_entity_order(dir_triples=dir_triples, name_texts=name_texts)

    observed: set[str] = set()
    for split in ("train.txt", "valid.txt", "test.txt"):
        split_path = dir_triples / split
        if not split_path.exists() or split_path.stat().st_size == 0:
            continue
        for h, _, t in read_triples(split_path):
            observed.add(h)
            observed.add(t)

    if observed:
        # Keep stable ordering: existing list order first, then append new ones.
        seen = set(entities)
        for e in sorted(observed):
            if e not in seen:
                entities.append(e)
                seen.add(e)

    missing_entities = [e for e in entities if e not in entity_to_row]
    if missing_entities:
        logger.info("KG-FIT cache missing %d entities (extending). Example: %s", len(missing_entities), missing_entities[:3])

        precompute_config = precompute_config or KGFitPrecomputeConfig()
        model = str(meta.get("model") or precompute_config.model)
        embed_fn = build_embedder(model)

        names, descs = _ensure_complete_texts(
            entities=missing_entities,
            name_texts=name_texts,
            desc_texts=desc_texts,
            use_name_as_desc_if_missing=precompute_config.use_name_as_desc_if_missing,
        )
        new_name = embed_texts(
            texts=names,
            embed_fn=embed_fn,
            batch_size=int(precompute_config.batch_size),
        )
        new_desc = embed_texts(
            texts=descs,
            embed_fn=embed_fn,
            batch_size=int(precompute_config.batch_size),
        )

        # match dtype with existing cache
        new_name = new_name.astype(name_emb.dtype, copy=False)
        new_desc = new_desc.astype(desc_emb.dtype, copy=False)

        start = int(name_emb.shape[0])
        name_emb = np.concatenate([name_emb, new_name], axis=0)
        desc_emb = np.concatenate([desc_emb, new_desc], axis=0)
        for i, entity in enumerate(missing_entities):
            entity_to_row[entity] = start + i

        np.save(name_path, name_emb)
        np.save(desc_path, desc_emb)
        meta["entity_to_row"] = entity_to_row
        meta["created_at"] = meta.get("created_at") or ""
        meta["updated_at"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S")
        _save_json(meta_path, meta)

        logger.info("Extended KG-FIT embeddings: rows=%d -> %d", start, int(name_emb.shape[0]))

    # --- seed hierarchy: extend labels for any missing entities
    hierarchy_path = cache_dir / "hierarchy_seed.json"
    centers_path = cache_dir / "cluster_embeddings.npy"
    neighbors_path = cache_dir / "neighbor_clusters.json"
    if not (hierarchy_path.exists() and centers_path.exists() and neighbors_path.exists()):
        logger.warning(
            "KG-FIT seed hierarchy artifacts missing under %s; skipping hierarchy extension.",
            cache_dir,
        )
        return

    hierarchy = _load_json(hierarchy_path)
    entity_ids: List[str] = list(hierarchy.get("entity_ids") or [])
    labels: List[int] = list(hierarchy.get("labels") or [])
    cluster_labels: List[int] = [int(x) for x in hierarchy.get("cluster_labels") or []]
    if not entity_ids or not labels:
        raise ValueError(f"Invalid hierarchy_seed.json: missing entity_ids/labels: {hierarchy_path}")
    if len(entity_ids) != len(labels):
        raise ValueError(
            f"Invalid hierarchy_seed.json: len(entity_ids)={len(entity_ids)} != len(labels)={len(labels)}"
        )
    if not cluster_labels:
        cluster_labels = sorted(set(int(x) for x in labels))
        hierarchy["cluster_labels"] = cluster_labels

    existing = set(entity_ids)
    missing_h = [e for e in entities if e not in existing]
    if not missing_h:
        return

    centers = np.load(centers_path)
    if centers.ndim != 2:
        raise ValueError(f"Invalid cluster_embeddings.npy shape={centers.shape}")
    if centers.shape[0] != len(cluster_labels):
        logger.warning(
            "cluster_centers rows (%d) != cluster_labels (%d); proceeding with min.",
            int(centers.shape[0]),
            len(cluster_labels),
        )
        min_k = min(int(centers.shape[0]), len(cluster_labels))
        centers = centers[:min_k]
        cluster_labels = cluster_labels[:min_k]
        hierarchy["cluster_labels"] = cluster_labels

    rows = [entity_to_row[e] for e in missing_h]
    miss_name = name_emb[rows]
    miss_desc = desc_emb[rows]
    merged = _reshape_merged_embeddings(
        name_emb=miss_name,
        desc_emb=miss_desc,
        reshape_strategy=reshape_strategy,
        embedding_dim=embedding_dim,
    )
    sims = _cosine_sim_to_centers(merged, centers)
    nearest = np.argmax(sims, axis=1)

    for entity, center_idx in zip(missing_h, nearest.tolist()):
        entity_ids.append(entity)
        labels.append(int(cluster_labels[int(center_idx)]))

    hierarchy["entity_ids"] = entity_ids
    hierarchy["labels"] = labels
    _save_json(hierarchy_path, hierarchy)

    logger.info(
        "Extended KG-FIT hierarchy labels: +%d entities (total=%d)",
        len(missing_h),
        len(entity_ids),
    )
