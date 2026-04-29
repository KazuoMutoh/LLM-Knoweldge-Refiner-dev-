#!/usr/bin/env python3
"""Compute KG-FIT text embeddings from entity2text files."""

from __future__ import annotations

import argparse
from pathlib import Path

from simple_active_refine.kgfit_precompute import KGFitPrecomputeConfig, precompute_kgfit_embeddings
from simple_active_refine.util import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute KG-FIT text embeddings")
    p.add_argument("--dir_triples", required=True, help="Dataset directory")
    p.add_argument("--output_dir", default=None, help="Output dir (default: <dir_triples>/.cache/kgfit)")
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    p.add_argument("--name_source", default="entity2text.txt")
    p.add_argument("--desc_source", default="entity2textlong.txt")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--use_name_as_desc_if_missing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = KGFitPrecomputeConfig(
        model=args.model,
        batch_size=args.batch_size,
        dtype=args.dtype,
        use_name_as_desc_if_missing=args.use_name_as_desc_if_missing,
    )
    precompute_kgfit_embeddings(
        dir_triples=Path(args.dir_triples),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        name_source=args.name_source,
        desc_source=args.desc_source,
        config=cfg,
        max_items=args.max_items,
    )
    logger.info("KG-FIT text embedding precompute finished")


if __name__ == "__main__":
    main()
