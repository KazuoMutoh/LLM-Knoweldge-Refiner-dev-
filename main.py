"""
Main script (v3) using pipeline abstractions.

This version reuses the pipeline interfaces in simple_active_refine.pipeline to
structure rule extraction, triple acquisition, evaluation, and final KGE
training. It keeps the high-score AMIE+ rule extraction and diverse selection
from main_v2_revised, but delegates the loop to RuleDrivenKGRefinementPipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict

from simple_active_refine.pipeline import RefinedKG, RuleDrivenKGRefinementPipeline, RuleExtractionContext
from simple_active_refine.data_manager import IterationDataManager, load_triples
from simple_active_refine.rule_extractor_impl import HighScoreRuleExtractor, PrecomputedRuleExtractor
from simple_active_refine.triple_acquirer_impl import RuleBasedTripleAcquirer, RandomTripleAcquirer
from simple_active_refine.kge_trainer_impl import FinalKGETrainer
from simple_active_refine.rule_selector import create_rule_selector
from simple_active_refine.rule_history import RuleHistory
from simple_active_refine.util import get_logger

logger = get_logger("main_v3")


def save_markdown(md_text: str, file_path: str) -> None:
    """Save Markdown text to a file (simple helper for reports/tests)."""

    with open(file_path, "w", encoding="utf-8") as fout:
        fout.write(md_text)


def sample_target_triples(all_target_triples, n_sample, exclude_triples=None):
    """Randomly sample target triples with optional exclusion."""

    if exclude_triples is None:
        exclude_triples = set()

    available = [t for t in all_target_triples if t not in exclude_triples]
    n_sample = min(n_sample, len(available))
    return random.sample(available, n_sample)


def main():
    # Parse CLI options controlling iteration count, rule mining, and embeddings
    parser = argparse.ArgumentParser(description="Knowledge Graph Improvement (v3 pipeline)")
    parser.add_argument("--n_iter", type=int, default=2, help="Number of refinement rounds")
    parser.add_argument("--num_epochs", type=int, default=2, help="Embedding epochs per round")
    parser.add_argument("--dir", type=str, default="./experiments/20251215/v3_run", help="Working directory")
    parser.add_argument(
        "--dir_initial_triples",
        type=str,
        default="./experiments/test_data_for_nationality_v3",
        help="Initial dataset directory containing train/valid/test",
    )
    parser.add_argument("--use_high_score_triples", action="store_true", help="Use high-score triples for rule mining")
    parser.add_argument("--lower_percentile", type=float, default=80.0, help="Lower percentile for high-score mining")
    parser.add_argument("--k_neighbor", type=int, default=1, help="K-hop neighborhood for subgraph extraction")
    parser.add_argument("--min_head_coverage", type=float, default=0.01, help="Minimum head coverage for AMIE+")
    parser.add_argument("--min_pca_conf", type=float, default=0.05, help="Minimum PCA confidence for AMIE+")
    parser.add_argument("--use_llm_rule_filter", action="store_true", help="Filter extracted rules with LLMRuleFilter")
    parser.add_argument("--llm_model", type=str, default="gpt-4o", help="LLM model for rule filtering")
    parser.add_argument("--llm_temperature", type=float, default=0.0, help="LLM temperature for rule filtering")
    parser.add_argument("--llm_request_timeout", type=int, default=120, help="LLM timeout (sec) for rule filtering")
    parser.add_argument("--llm_max_tokens", type=int, default=16000, help="Max tokens for LLM rule filtering output")
    parser.add_argument("--llm_top_k", type=int, default=None, help="Top-K rules to keep after LLM filtering (defaults to n_rules_pool)")
    parser.add_argument("--llm_min_pca_conf", type=float, default=None, help="Optional PCA conf floor before LLM scoring")
    parser.add_argument("--llm_min_head_coverage", type=float, default=None, help="Optional head coverage floor before LLM scoring")
    parser.add_argument("--n_rules_pool", type=int, default=12, help="Number of rules to keep in the pool")
    parser.add_argument("--n_rules_select", type=int, default=3, help="Number of rules to select per iteration (adaptive)")
    parser.add_argument("--rule_selector_strategy", type=str, default="ucb", help="Rule selector strategy: ucb|epsilon_greedy|llm_policy")
    parser.add_argument("--n_targets_per_rule", type=int, default=10, help="Target triples sampled per rule")
    parser.add_argument("--use_random_acquirer", action="store_true", help="Use random triple acquirer instead of rule-based")
    parser.add_argument("--config_embedding", type=str, default="./config_embeddings.json", help="Embedding config path")
    args = parser.parse_args()

    # Load embedding config and override epochs from CLI
    with open(args.config_embedding, "r") as fin:
        config_embedding: Dict = json.load(fin)
    config_embedding["training_kwargs"]["num_epochs"] = args.num_epochs

    # Prepare data manager and initial KG from template triples
    template_dir = args.dir_initial_triples
    data_manager = IterationDataManager(template_dir=template_dir, working_dir=args.dir)

    train_path = os.path.join(template_dir, "train.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.txt not found in {template_dir}")
    initial_triples = load_triples(train_path)
    initial_kg = RefinedKG(triples=initial_triples)

    target_path = os.path.join(template_dir, "target_triples.txt")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"target_triples.txt not found in {template_dir}")
    target_triples = load_triples(target_path)

    logger.info("[v3] Building initial rule pool before iterations")
    # Mine AMIE+ rules (optionally filtered by LLM) to seed the rule pool
    pool_builder = HighScoreRuleExtractor(
        data_manager=data_manager,
        embedding_config=config_embedding,
        target_relation="/people/person/nationality",
        use_high_score_triples=args.use_high_score_triples,
        lower_percentile=args.lower_percentile,
        k_neighbor=args.k_neighbor,
        min_head_coverage=args.min_head_coverage,
        min_pca_conf=args.min_pca_conf,
        n_rules_pool=args.n_rules_pool,
        tmp_dir=os.path.join(args.dir, "tmp"),
        use_llm_filter=args.use_llm_rule_filter,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_request_timeout=args.llm_request_timeout,
        llm_max_tokens=args.llm_max_tokens,
        llm_top_k=args.llm_top_k,
        llm_min_pca_conf=args.llm_min_pca_conf,
        llm_min_head_coverage=args.llm_min_head_coverage,
    )
    pool_result = pool_builder.extract(RuleExtractionContext(kg=initial_kg, iteration=0))
    extractor = PrecomputedRuleExtractor(rules=pool_result.rules, diagnostics=pool_result.diagnostics)

    # Choose how to pick candidate triples for each rule
    if args.use_random_acquirer:
        acquirer = RandomTripleAcquirer(
            target_triples=target_triples,
            n_targets_per_rule=args.n_targets_per_rule,
            reuse_targets=False,
            dump_base_dir=args.dir,
        )
    else:
        acquirer = RuleBasedTripleAcquirer(
            target_triples=target_triples,
            candidate_dir=template_dir,
            n_targets_per_rule=args.n_targets_per_rule,
            dump_base_dir=args.dir,
        )

    # Candidate acquisition is treated as implicit acceptance by default.
    # Provide an explicit evaluator only when you want filtering or richer scoring.
    evaluator = None
    kge_trainer = FinalKGETrainer(data_manager=data_manager, embedding_config=config_embedding)

    # Track rule performance and create selector for adaptive choice per round
    rule_history = RuleHistory()
    selector = create_rule_selector(strategy=args.rule_selector_strategy, history=rule_history)

    # Wire all components into the iterative refinement pipeline
    pipeline = RuleDrivenKGRefinementPipeline(
        rule_extractor=extractor,
        triple_acquirer=acquirer,
        triple_evaluator=evaluator,
        kge_trainer=kge_trainer,
        rule_selector=selector,
        rule_history=rule_history,
        n_select_rules=args.n_rules_select,
    )

    logger.info("[v3] Starting pipeline run with precomputed rule pool (%d rules)", len(pool_result.rules))
    # Run refinement loop: select rules, acquire/evaluate triples, retrain KGE, track metrics
    result = pipeline.run(initial_kg=initial_kg, num_rounds=args.n_iter, kge_output_dir=os.path.join(args.dir, "final"))
    logger.info("[v3] Pipeline finished")

    # Persist final KG and log summary stats
    final_dir = data_manager.write_custom("final_dataset", result.final_kg)
    logger.info("[v3] Final KG written to %s", final_dir)
    logger.info("[v3] Rounds: %d", len(result.rounds))
    if result.kge_result:
        logger.info(
            "[v3] Final KGE metrics: Hits@1=%.4f, Hits@3=%.4f, Hits@10=%.4f, MRR=%.4f",
            result.kge_result.hits_at_1,
            result.kge_result.hits_at_3,
            result.kge_result.hits_at_10,
            result.kge_result.mrr,
        )


if __name__ == "__main__":
    main()
