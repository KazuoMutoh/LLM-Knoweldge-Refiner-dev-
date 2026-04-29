#!/usr/bin/env python
"""Build an initial rule pool for a target relation using a trained KGE model.

Inputs
- Trained KGE directory (dir containing trained_model.pkl and training_triples/)
- Target relation path (e.g., /people/person/nationality)

Outputs (written under --output-dir)
- initial_rule_pool.csv  : Rules with AMIE+ metrics
- initial_rule_pool.pkl  : Pickled AmieRules object
- initial_rule_pool.txt  : Human-readable summary

This script mines AMIE+ rules from the KG defined by the embedding, then
selects the top-N rules to form the initial rule pool. It avoids LLM calls by
reusing the AMIE-derived rules directly.
"""

import argparse
from pathlib import Path
from typing import Tuple

from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.rule_extractor import (
    extract_rules_from_entire_graph,
    extract_rules_from_high_score_triples,
)
from simple_active_refine.rule_generator import BaseRuleGenerator
from simple_active_refine.util import get_logger

logger = get_logger("build_initial_rule_pool")


def _summarize_rule_pool(rule_pool) -> str:
    """Create a compact text summary of the rule pool."""
    lines = []
    for idx, rule in enumerate(rule_pool.rules, 1):
        body = " ; ".join(tp.to_tsv() for tp in rule.body)
        support = f"{rule.support:.4f}" if rule.support is not None else "-"
        pca_conf = f"{rule.pca_conf:.4f}" if rule.pca_conf is not None else "-"
        head_cov = f"{rule.head_coverage:.4f}" if rule.head_coverage is not None else "-"
        lines.append(
            f"[{idx:02d}] head={rule.head.to_tsv()} | body={body} | support={support} | pca_conf={pca_conf} | head_coverage={head_cov}"
        )
    return "\n".join(lines)


def build_initial_rule_pool(
    model_dir: str,
    target_relation: str,
    output_dir: str,
    n_rules: int,
    sort_by: str,
    mode: str,
    min_head_coverage: float,
    min_pca_conf: float,
    lower_percentile: float,
    k_neighbor: int,
) -> Tuple[str, str, str]:
    """Generate the initial rule pool and save artifacts.

    Returns tuple of (csv_path, pkl_path, summary_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading KGE model from %s", model_dir)
    kge = KnowledgeGraphEmbedding(model_dir=model_dir)

    amie_work_dir = out_dir / "amie_tmp"
    amie_work_dir.mkdir(parents=True, exist_ok=True)

    if mode == "entire":
        logger.info("Extracting AMIE+ rules from the entire graph")
        amie_rules = extract_rules_from_entire_graph(
            kge=kge,
            target_relation=target_relation,
            top_k=None,
            sorted_by=None,
            min_head_coverage=min_head_coverage,
            min_pca_conf=min_pca_conf,
            dir_working=str(amie_work_dir),
        )
    else:
        logger.info(
            "Extracting AMIE+ rules from high-score triples (percentile >= %.1f, %d-hop)",
            lower_percentile,
            k_neighbor,
        )
        amie_rules = extract_rules_from_high_score_triples(
            kge=kge,
            target_relation=target_relation,
            top_k=None,
            lower_percentile=lower_percentile,
            k_neighbor=k_neighbor,
            sorted_by=None,
            min_head_coverage=min_head_coverage,
            min_pca_conf=min_pca_conf,
            dir_working=str(amie_work_dir),
        )

    if not amie_rules.rules:
        raise RuntimeError("AMIE+ did not return any rules. Check inputs and thresholds.")

    generator = BaseRuleGenerator()
    rule_pool = generator.create_initial_rule_pool_from_amie(
        amie_rules=amie_rules,
        n_rules=n_rules,
        sort_by=sort_by,
    )

    csv_path = out_dir / "initial_rule_pool.csv"
    pkl_path = out_dir / "initial_rule_pool.pkl"
    summary_path = out_dir / "initial_rule_pool.txt"

    rule_pool.to_csv(str(csv_path))
    rule_pool.to_pickle(str(pkl_path))
    summary_path.write_text(_summarize_rule_pool(rule_pool), encoding="utf-8")

    logger.info("Saved initial rule pool: %s", csv_path)
    logger.info("Saved pickle: %s", pkl_path)
    logger.info("Saved summary: %s", summary_path)

    return str(csv_path), str(pkl_path), str(summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build initial rule pool using AMIE+ and a trained KGE model.")
    parser.add_argument("--model-dir", required=True, help="Path to trained KGE directory (contains trained_model.pkl)")
    parser.add_argument("--target-relation", required=True, help="Target relation path (e.g., /people/person/nationality)")
    parser.add_argument("--output-dir", default="./tmp/initial_rule_pool", help="Directory to store outputs")
    parser.add_argument("--n-rules", type=int, default=20, help="Number of rules to keep in the initial pool")
    parser.add_argument("--sort-by", default="support", help="Metric column used to rank AMIE+ rules (e.g., support, pca_conf)")
    parser.add_argument(
        "--mode",
        choices=["entire", "high-score"],
        default="entire",
        help="Choose entire-graph AMIE or high-score target triples extraction",
    )
    parser.add_argument("--min-head-coverage", type=float, default=0.01, help="Minimum head coverage for AMIE+")
    parser.add_argument("--min-pca-conf", type=float, default=0.01, help="Minimum PCA confidence for AMIE+")
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=90.0,
        help="Percentile threshold for selecting high-score triples (only for high-score mode)",
    )
    parser.add_argument("--k-neighbor", type=int, default=1, help="Hop size for enclosing subgraph (only for high-score mode)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_initial_rule_pool(
        model_dir=args.model_dir,
        target_relation=args.target_relation,
        output_dir=args.output_dir,
        n_rules=args.n_rules,
        sort_by=args.sort_by,
        mode=args.mode,
        min_head_coverage=args.min_head_coverage,
        min_pca_conf=args.min_pca_conf,
        lower_percentile=args.lower_percentile,
        k_neighbor=args.k_neighbor,
    )
