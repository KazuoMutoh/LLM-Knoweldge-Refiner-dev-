#!/usr/bin/env python3
"""Build KG-FIT seed hierarchy from precomputed embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simple_active_refine.kgfit import (
    KGFitEmbeddingConfig,
    KGFitEmbeddingPaths,
    load_kgfit_raw_embeddings,
    resolve_kgfit_paths,
    KGFitEmbeddingError,
)
from simple_active_refine.kgfit_hierarchy import build_seed_hierarchy, compute_neighbor_clusters
from simple_active_refine.util import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build KG-FIT seed hierarchy")
    p.add_argument("--dir_triples", required=True, help="Dataset directory")
    p.add_argument("--output_dir", default=None, help="Output directory (default: <dir_triples>/.cache/kgfit)")
    p.add_argument("--name_embeddings", default=None, help="Path to name embeddings (.npy)")
    p.add_argument("--desc_embeddings", default=None, help="Path to description embeddings (.npy)")
    p.add_argument("--meta", default=None, help="Path to meta json with entity_to_row")
    p.add_argument("--reshape_strategy", default="full", choices=["full", "slice"], help="Embedding reshape strategy")
    p.add_argument("--embedding_dim", type=int, default=None, help="Embedding dim for slice strategy")
    p.add_argument("--tau_min", type=float, default=0.15)
    p.add_argument("--tau_max", type=float, default=0.85)
    p.add_argument("--tau_steps", type=int, default=15)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--neighbor_k", type=int, default=5)
    p.add_argument("--random_seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    dir_triples = Path(args.dir_triples)
    output_dir = Path(args.output_dir) if args.output_dir else (dir_triples / ".cache" / "kgfit")
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    if args.name_embeddings:
        overrides["name_embeddings"] = args.name_embeddings
    if args.desc_embeddings:
        overrides["desc_embeddings"] = args.desc_embeddings
    if args.meta:
        overrides["meta"] = args.meta

    paths = resolve_kgfit_paths(dir_triples=dir_triples, override=overrides if overrides else None)

    try:
        name_embeddings, desc_embeddings, entity_ids = load_kgfit_raw_embeddings(paths=paths)
    except KGFitEmbeddingError as err:
        raise SystemExit(f"KG-FIT embeddings missing: {err}") from err

    config = KGFitEmbeddingConfig(
        paths=paths,
        reshape_strategy=args.reshape_strategy,
        embedding_dim=args.embedding_dim,
    )

    if config.reshape_strategy == "full":
        merged = np.concatenate([name_embeddings, desc_embeddings], axis=1)
    elif config.reshape_strategy == "slice":
        if config.embedding_dim is None or config.embedding_dim % 2 != 0:
            raise SystemExit("embedding_dim must be even when using slice strategy")
        half = config.embedding_dim // 2
        merged = np.concatenate([name_embeddings[:, :half], desc_embeddings[:, :half]], axis=1)
    else:
        raise SystemExit(f"Unsupported reshape_strategy: {config.reshape_strategy}")

    result = build_seed_hierarchy(
        embeddings=merged,
        entity_ids=entity_ids,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_steps=args.tau_steps,
        max_silhouette_samples=args.max_samples,
        random_seed=args.random_seed,
    )

    hierarchy_path = output_dir / "hierarchy_seed.json"
    with hierarchy_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tau_opt": result.tau_opt,
                "tau_min": args.tau_min,
                "tau_max": args.tau_max,
                "tau_steps": args.tau_steps,
                "labels": result.labels.tolist(),
                "clusters": result.clusters,
                "cluster_labels": result.cluster_labels,
                "entity_ids": list(entity_ids),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    centers_path = output_dir / "cluster_embeddings.npy"
    np.save(centers_path, result.cluster_centers)

    neighbors = compute_neighbor_clusters(
        cluster_centers=result.cluster_centers,
        k_neighbors=args.neighbor_k,
    )
    neighbors_path = output_dir / "neighbor_clusters.json"
    neighbors_path.write_text(json.dumps(neighbors, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Saved seed hierarchy: %s", hierarchy_path)
    logger.info("Saved cluster centers: %s", centers_path)
    logger.info("Saved neighbor clusters: %s", neighbors_path)


if __name__ == "__main__":
    main()
