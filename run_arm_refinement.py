#!/usr/bin/env python3
"""Run arm-driven iterative refinement (v1).

This CLI consumes:
- initial_arms.json/pkl from build_initial_arms.py
- initial_rule_pool.pkl to resolve arm rule keys
- triples directory containing train.txt
- target_triples.txt
- candidate triples file (e.g., train_removed.txt)

and writes per-iteration outputs under:
    base_output_path/iter_k/

Notes:
    This CLI supports two candidate acquisition modes:
    - local: use candidate_triples (e.g., train_removed.txt)
    - web: retrieve candidates via LLM/web search per iteration
"""

from __future__ import annotations

import argparse
from typing import Optional

from simple_active_refine.arm_pipeline import ArmDrivenKGRefinementPipeline, ArmPipelineConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run arm-driven KG refinement (evidence-first, store-only hypothesis)")
    p.add_argument("--base_output_path", required=True, help="Base output directory (iter_k will be created under this)")

    p.add_argument("--initial_arms", required=True, help="Path to initial_arms.json or initial_arms.pkl")
    p.add_argument("--rule_pool_pkl", required=True, help="Path to initial_rule_pool.pkl (AmieRules)")

    p.add_argument("--dir_triples", required=True, help="Directory containing train.txt")
    p.add_argument("--target_triples", required=True, help="Path to target_triples.txt")
    p.add_argument("--candidate_triples", required=True, help="Path to candidate triples (e.g., train_removed.txt)")

    p.add_argument("--n_iter", type=int, default=1)
    p.add_argument("--k_sel", type=int, default=3)
    p.add_argument("--n_targets_per_arm", type=int, default=50)
    p.add_argument("--max_witness_per_head", type=int, default=None)

    p.add_argument(
        "--candidate_source",
        default="local",
        choices=["local", "web"],
        help=(
            "Candidate source for evidence acquisition: local (candidate_triples) or web (LLM/web search). "
            "Note: candidate_triples is still used for incident triple augmentation."
        ),
    )
    p.add_argument("--web_llm_model", default="gpt-4o", help="LLM model for web retrieval (candidate_source=web)")
    p.add_argument(
        "--disable_web_search",
        action="store_true",
        help="Disable OpenAI web_search_preview tool and use plain LLM only (candidate_source=web)",
    )
    p.add_argument(
        "--web_max_targets_total_per_iteration",
        type=int,
        default=20,
        help="Cap on total (arm,target) pairs to query per iteration (candidate_source=web)",
    )
    p.add_argument(
        "--web_max_triples_per_iteration",
        type=int,
        default=200,
        help="Cap on number of web-retrieved candidate triples kept per iteration (candidate_source=web)",
    )
    p.add_argument(
        "--disable_entity_linking",
        action="store_true",
        help="Disable entity linking to match web entities with existing KG entities (candidate_source=web)",
    )

    p.add_argument("--selector_strategy", default="ucb", choices=["ucb", "epsilon_greedy", "llm_policy", "random"])
    p.add_argument("--selector_exploration_param", type=float, default=1.0)
    p.add_argument("--selector_epsilon", type=float, default=0.1)

    p.add_argument("--witness_weight", type=float, default=1.0)
    p.add_argument("--evidence_weight", type=float, default=1.0)

    p.add_argument(
        "--disable_incident_triples",
        action="store_true",
        help=(
            "Disable incident triple augmentation for newly introduced entities. "
            "By default, the pipeline may add extra candidate triples incident to new entities to avoid dangling entities."
        ),
    )
    p.add_argument(
        "--max_incident_candidate_triples_per_iteration",
        type=int,
        default=None,
        help=(
            "Optional cap for incident triples per iteration (after filtering). "
            "If omitted, no cap is applied. Ignored when --disable_incident_triples is set."
        ),
    )

    p.add_argument(
        "--relation_priors_path",
        default=None,
        help=(
            "Optional JSON mapping predicate->prior (or predicate->{X,...}). "
            "If omitted, auto-detect <dir_triples>/relation_priors.json when present."
        ),
    )

    p.add_argument(
        "--disable_relation_priors",
        action="store_true",
        help=(
            "Disable relation priors (X_r) entirely, including auto-detect of <dir_triples>/relation_priors.json. "
            "Use this to enforce priors=off."
        ),
    )

    return p.parse_args()


def run(
    *,
    base_output_path: str,
    initial_arms: str,
    rule_pool_pkl: str,
    dir_triples: str,
    target_triples: str,
    candidate_triples: str,
    n_iter: int = 1,
    k_sel: int = 3,
    n_targets_per_arm: int = 50,
    max_witness_per_head: Optional[int] = None,
    candidate_source: str = "local",
    web_llm_model: str = "gpt-4o",
    disable_web_search: bool = False,
    web_max_targets_total_per_iteration: int = 20,
    web_max_triples_per_iteration: int = 200,
    disable_entity_linking: bool = False,
    selector_strategy: str = "ucb",
    selector_exploration_param: float = 1.0,
    selector_epsilon: float = 0.1,
    witness_weight: float = 1.0,
    evidence_weight: float = 1.0,
    disable_incident_triples: bool = False,
    max_incident_candidate_triples_per_iteration: Optional[int] = None,
    relation_priors_path: Optional[str] = None,
    disable_relation_priors: bool = False,
) -> None:
    """Run arm-driven refinement without argparse.

    This is the import-safe core entry point. The CLI `main()` delegates here.
    """

    cfg = ArmPipelineConfig(
        base_output_path=base_output_path,
        n_iter=n_iter,
        k_sel=k_sel,
        n_targets_per_arm=n_targets_per_arm,
        max_witness_per_head=max_witness_per_head,
        candidate_source=candidate_source,
        web_llm_model=web_llm_model,
        web_use_web_search=not bool(disable_web_search),
        web_max_targets_total_per_iteration=int(web_max_targets_total_per_iteration),
        web_max_triples_per_iteration=int(web_max_triples_per_iteration),
        web_enable_entity_linking=not bool(disable_entity_linking),
        selector_strategy=selector_strategy,
        selector_exploration_param=selector_exploration_param,
        selector_epsilon=selector_epsilon,
        witness_weight=witness_weight,
        evidence_weight=evidence_weight,
        add_incident_candidate_triples_for_new_entities=not bool(disable_incident_triples),
        max_incident_candidate_triples_per_iteration=max_incident_candidate_triples_per_iteration,
        relation_priors_path=relation_priors_path,
        disable_relation_priors=bool(disable_relation_priors),
    )

    pipe = ArmDrivenKGRefinementPipeline.from_paths(
        config=cfg,
        initial_arms_path=initial_arms,
        rule_pool_pkl=rule_pool_pkl,
        dir_triples=dir_triples,
        target_triples_path=target_triples,
        candidate_triples_path=candidate_triples,
    )
    pipe.run()


def main() -> None:
    args = parse_args()

    run(
        base_output_path=args.base_output_path,
        initial_arms=args.initial_arms,
        rule_pool_pkl=args.rule_pool_pkl,
        dir_triples=args.dir_triples,
        target_triples=args.target_triples,
        candidate_triples=args.candidate_triples,
        n_iter=args.n_iter,
        k_sel=args.k_sel,
        n_targets_per_arm=args.n_targets_per_arm,
        max_witness_per_head=args.max_witness_per_head,
        candidate_source=args.candidate_source,
        web_llm_model=args.web_llm_model,
        disable_web_search=bool(args.disable_web_search),
        web_max_targets_total_per_iteration=args.web_max_targets_total_per_iteration,
        web_max_triples_per_iteration=args.web_max_triples_per_iteration,
        disable_entity_linking=bool(args.disable_entity_linking),
        selector_strategy=args.selector_strategy,
        selector_exploration_param=args.selector_exploration_param,
        selector_epsilon=args.selector_epsilon,
        witness_weight=args.witness_weight,
        evidence_weight=args.evidence_weight,
        disable_incident_triples=args.disable_incident_triples,
        max_incident_candidate_triples_per_iteration=args.max_incident_candidate_triples_per_iteration,
        relation_priors_path=args.relation_priors_path,
        disable_relation_priors=bool(args.disable_relation_priors),
    )


if __name__ == "__main__":
    main()
