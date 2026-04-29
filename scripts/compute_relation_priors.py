#!/usr/bin/env python3
"""Compute relation priors (X_r) for witness weighting.

This script computes X_r(2), X_r(3), X_r(4), X_r(7) and aggregates them into a
final X_r via weighted sum, then writes a JSON file.

By default, it reads train triples from --dataset_dir/train.txt and reads entity
embeddings from --model_before_dir (PyKEEN saved run directory).

Output format is a payload:
  {"meta": {...}, "priors": {predicate: {"X":..., "X2":..., ...}}}

Note: The arm pipeline loader expects a simpler mapping predicate -> X (or
predicate -> {"X": X}). If you want to use this payload directly with the
existing loader, pass a path and use "priors" object as the root, or extend the
loader. (We keep the payload for traceability.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_active_refine.relation_priors_compute import RelationPriorConfig, compute_and_save_relation_priors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute relation priors (KGE-friendly X_r)")
    p.add_argument("--dataset_dir", required=True, help="Dataset directory containing train.txt")
    p.add_argument("--model_before_dir", default=None, help="Trained KGE directory (contains trained_model.pkl)")
    p.add_argument(
        "--output_path",
        default=None,
        help="Output JSON path (default: <dataset_dir>/relation_priors.json)",
    )

    p.add_argument("--max_samples_x3_per_relation", type=int, default=2000)
    p.add_argument("--max_samples_x7_per_relation", type=int, default=5000)
    p.add_argument("--min_count_x7", type=int, default=50)
    p.add_argument("--random_seed", type=int, default=0)

    p.add_argument("--weight_x2", type=float, default=0.0)
    p.add_argument("--weight_x3", type=float, default=0.0)
    p.add_argument("--weight_x4", type=float, default=0.0)
    p.add_argument("--weight_x7", type=float, default=1.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output_path) if args.output_path else (dataset_dir / "relation_priors.json")

    cfg = RelationPriorConfig(
        max_samples_x3_per_relation=int(args.max_samples_x3_per_relation),
        max_samples_x7_per_relation=int(args.max_samples_x7_per_relation),
        random_seed=int(args.random_seed),
        min_count_x7=int(args.min_count_x7),
        weight_x2=float(args.weight_x2),
        weight_x3=float(args.weight_x3),
        weight_x4=float(args.weight_x4),
        weight_x7=float(args.weight_x7),
    )

    compute_and_save_relation_priors(
        dataset_dir=str(dataset_dir),
        model_before_dir=args.model_before_dir,
        output_path=str(output_path),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
