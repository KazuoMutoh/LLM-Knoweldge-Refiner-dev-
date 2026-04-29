#!/usr/bin/env python
"""Build initial arm pool from a rule pool and target/candidate triples.

- Input rule pool: initial_rule_pool.pkl (AmieRules) from build_initial_rule_pool.py
- Input triples: target_triples.txt (heads to test), candidate triples (train/train_removed)
- Output: arms.json, arms.pkl, arms_summary.txt
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

from simple_active_refine.amie import AmieRules
from simple_active_refine.arm_builder import ArmBuilderConfig, build_initial_arms, save_arms_json
from simple_active_refine.util import get_logger

logger = get_logger("build_initial_arms")

Triple = Tuple[str, str, str]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON config must be an object, got {type(obj)}")
    return obj


def _load_triples(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def _summarize(arms) -> str:
    lines = []
    for idx, arm in enumerate(arms, 1):
        lines.append(f"[{idx:02d}] type={arm.arm_type} rules={arm.rule_keys} meta={arm.metadata}")
    return "\n".join(lines)


def _load_candidates_from_dir(dir_triples: str, include_train_removed: bool = True) -> List[Triple]:
    """Load candidate triples from a KG directory (train + optional train_removed)."""

    candidates: List[Triple] = []
    train_path = Path(dir_triples) / "train.txt"
    if train_path.exists():
        candidates.extend(_load_triples(str(train_path)))
    else:
        raise FileNotFoundError(f"train.txt not found in {dir_triples}")

    if include_train_removed:
        tr_path = Path(dir_triples) / "train_removed.txt"
        if tr_path.exists():
            candidates.extend(_load_triples(str(tr_path)))

    return candidates


def _load_train_from_dir(dir_triples: str) -> List[Triple]:
    """Load train triples (train.txt only) from a KG directory."""

    train_path = Path(dir_triples) / "train.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"train.txt not found in {dir_triples}")
    return _load_triples(str(train_path))


def build_initial_arm_pool(
    rule_pool_path: str,
    target_triples_path: str,
    candidate_triples_path: str | None,
    dir_triples: str | None,
    include_train_removed: bool,
    output_dir: str,
    k_pairs: int,
    max_witness_per_head: int | None,
    rule_top_k: int | None,
    rule_sort_by: str,
    min_pca_conf: float | None,
    min_head_coverage: float | None,
    exclude_relation_patterns: List[str] | None,
    rule_filter_config_path: str | None,
    pair_support_source: str = "candidate",
    pair_support_triples_path: str | None = None,
) -> tuple[str, str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading rule pool from %s", rule_pool_path)
    rule_pool = AmieRules.from_pickle(rule_pool_path)

    # Optional: prefilter rule pool to reduce singleton arm count.
    if rule_filter_config_path:
        cfg = _load_json(rule_filter_config_path)
        rule_top_k = int(cfg.get("top_k", rule_top_k or 0)) if cfg.get("top_k") is not None else rule_top_k
        use_llm = bool(cfg.get("use_llm", False))
        min_head_coverage = cfg.get("min_head_coverage", min_head_coverage)
        min_pca_conf = cfg.get("min_pca_conf", min_pca_conf)
        rule_sort_by = "llm" if use_llm else rule_sort_by

    rules_before = len(rule_pool.rules)
    if exclude_relation_patterns:
        rule_pool = rule_pool.exclude_relations_by_pattern(exclude_relation_patterns)

    if rule_top_k is not None:
        rule_pool = rule_pool.filter(
            min_pca_conf=min_pca_conf,
            min_head_coverage=min_head_coverage,
            sort_by=rule_sort_by,
            top_k=rule_top_k,
        )
    else:
        if min_pca_conf is not None or min_head_coverage is not None:
            rule_pool = rule_pool.filter(
                min_pca_conf=min_pca_conf,
                min_head_coverage=min_head_coverage,
                sort_by=rule_sort_by,
                top_k=len(rule_pool.rules),
            )

    logger.info(
        "Rule pool prefilter: before=%d after=%d (top_k=%s sort_by=%s min_pca_conf=%s min_head_coverage=%s exclude_patterns=%s)",
        rules_before,
        len(rule_pool.rules),
        str(rule_top_k) if rule_top_k is not None else "None",
        rule_sort_by,
        str(min_pca_conf) if min_pca_conf is not None else "None",
        str(min_head_coverage) if min_head_coverage is not None else "None",
        exclude_relation_patterns or [],
    )

    logger.info("Loading target triples from %s", target_triples_path)
    target_triples = _load_triples(target_triples_path)

    if pair_support_source not in {"candidate", "train"}:
        raise ValueError(
            f"Unsupported pair_support_source={pair_support_source!r} (expected 'candidate' or 'train')"
        )

    if candidate_triples_path:
        logger.info("Loading candidate triples from %s", candidate_triples_path)
        candidate_triples = _load_triples(candidate_triples_path)
    else:
        if not dir_triples:
            raise ValueError("Either --candidate-triples or --dir-triples must be provided")
        logger.info("Loading candidate triples from dir_triples=%s (train.txt%s)", dir_triples, 
                    " + train_removed.txt" if include_train_removed else "")
        candidate_triples = _load_candidates_from_dir(dir_triples, include_train_removed=include_train_removed)

    pair_support_triples: List[Triple] | None = None
    if pair_support_source == "train":
        if pair_support_triples_path:
            logger.info("Loading pair-support triples from %s", pair_support_triples_path)
            pair_support_triples = _load_triples(pair_support_triples_path)
        else:
            if not dir_triples:
                raise ValueError(
                    "pair_support_source='train' requires --dir-triples (to read train.txt) or --pair-support-triples"
                )
            logger.info("Loading pair-support triples from dir_triples=%s (train.txt only)", dir_triples)
            pair_support_triples = _load_train_from_dir(dir_triples)

    cfg = ArmBuilderConfig(
        k_pairs=k_pairs,
        max_witness_per_head=max_witness_per_head,
        pair_support_source=pair_support_source,
    )
    arms = build_initial_arms(
        rule_pool.rules,
        target_triples,
        candidate_triples,
        cfg,
        pair_support_triples=pair_support_triples,
    )

    json_path = out_dir / "initial_arms.json"
    pkl_path = out_dir / "initial_arms.pkl"
    summary_path = out_dir / "initial_arms.txt"

    save_arms_json(arms, str(json_path))
    with open(pkl_path, "wb") as f:
        pickle.dump(arms, f)
    summary_path.write_text(_summarize(arms), encoding="utf-8")

    logger.info("Saved arms: %s", json_path)
    logger.info("Saved pickle: %s", pkl_path)
    logger.info("Saved summary: %s", summary_path)

    return str(json_path), str(pkl_path), str(summary_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build initial arm pool from rule pool and triples")
    p.add_argument("--rule-pool", required=True, help="Path to initial_rule_pool.pkl")
    p.add_argument("--target-triples", required=True, help="Path to target_triples.txt")
    p.add_argument("--candidate-triples", required=False, help="Path to candidate triples (e.g., train.txt or merged)")
    p.add_argument("--dir-triples", required=False, help="Directory containing train.txt (and optionally train_removed.txt)")
    p.add_argument("--include-train-removed", action="store_true", help="When using --dir-triples, also load train_removed.txt if present")

    p.add_argument(
        "--pair-support-source",
        choices=["candidate", "train"],
        default="candidate",
        help=(
            "Which triple set to use when computing pair-arm support/Jaccard: "
            "candidate (default; backward compatible) or train (train.txt only)."
        ),
    )
    p.add_argument(
        "--pair-support-triples",
        default=None,
        help=(
            "Optional explicit triples file to use for pair-arm support/Jaccard. "
            "Used only when --pair-support-source=train. If omitted, uses <dir_triples>/train.txt."
        ),
    )
    p.add_argument("--output-dir", default="./tmp/initial_arms", help="Directory to store outputs")
    p.add_argument("--k-pairs", type=int, default=20, help="Top pair arms to keep by co-occurrence")
    p.add_argument("--max-witness-per-head", type=int, default=None, help="Optional cap for witness counting per head")

    # Rule prefilter options (to reduce singleton arm count).
    p.add_argument("--rule-top-k", type=int, default=None, help="Keep only top-K rules after filtering (reduces singleton arms)")
    p.add_argument(
        "--rule-sort-by",
        default="pca_conf",
        help="Sort key for selecting top rules: support|std_conf|pca_conf|head_coverage|body_size|pca_body_size|llm",
    )
    p.add_argument("--min-pca-conf", type=float, default=None, help="Optional minimum pca_conf threshold")
    p.add_argument("--min-head-coverage", type=float, default=None, help="Optional minimum head_coverage threshold")
    p.add_argument(
        "--exclude-relation-pattern",
        action="append",
        default=None,
        help="Exclude rules containing this substring in head/body relations (repeatable)",
    )
    p.add_argument(
        "--rule-filter-config",
        default=None,
        help="Optional JSON config compatible with config_rule_filter.json (top_k/use_llm/min_*)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_initial_arm_pool(
        rule_pool_path=args.rule_pool,
        target_triples_path=args.target_triples,
        candidate_triples_path=args.candidate_triples,
        dir_triples=args.dir_triples,
        include_train_removed=args.include_train_removed,
        output_dir=args.output_dir,
        k_pairs=args.k_pairs,
        max_witness_per_head=args.max_witness_per_head,
        rule_top_k=args.rule_top_k,
        rule_sort_by=args.rule_sort_by,
        min_pca_conf=args.min_pca_conf,
        min_head_coverage=args.min_head_coverage,
        exclude_relation_patterns=args.exclude_relation_pattern,
        rule_filter_config_path=args.rule_filter_config,
        pair_support_source=args.pair_support_source,
        pair_support_triples_path=args.pair_support_triples,
    )

