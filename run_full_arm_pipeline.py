#!/usr/bin/env python3
"""Run the full arm refinement pipeline end-to-end.

This script orchestrates the following existing CLI scripts in a single command:
  1) build_initial_rule_pool.py
  2) build_initial_arms.py
  3) run_arm_refinement.py (via simple_active_refine.arm_pipeline)
  4) retrain_and_evaluate_after_arm_run.py

It is designed to stabilize output directories under a single run directory and
support idempotent re-runs (skip if artifacts already exist), with an optional
--force to overwrite.

Notes
- The arm refinement step writes per-iteration outputs under <run_dir>/arm_run/iter_k/.
- The retrain/eval step prefers iter_*/accepted_added_triples.tsv (with fallback
    to accepted_evidence_triples.tsv for older runs) under the arm-run directory,
    and writes outputs under <arm_run_dir>/retrain_eval/.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from build_initial_arms import build_initial_arm_pool
from build_initial_rule_pool import build_initial_rule_pool
from retrain_and_evaluate_after_arm_run import run as retrain_and_evaluate
from simple_active_refine.arm_pipeline import ArmDrivenKGRefinementPipeline, ArmPipelineConfig
from simple_active_refine.relation_priors_compute import RelationPriorConfig, compute_and_save_relation_priors
from simple_active_refine.util import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RunnerPaths:
    run_dir: Path
    rule_pool_dir: Path
    arms_dir: Path
    arm_run_dir: Path

    rule_pool_pkl: Path
    arms_json: Path
    retrain_eval_summary: Path


def _resolve_paths(run_dir: str | Path) -> RunnerPaths:
    run_dir = Path(run_dir)
    rule_pool_dir = run_dir / "rule_pool"
    arms_dir = run_dir / "arms"
    arm_run_dir = run_dir / "arm_run"

    return RunnerPaths(
        run_dir=run_dir,
        rule_pool_dir=rule_pool_dir,
        arms_dir=arms_dir,
        arm_run_dir=arm_run_dir,
        rule_pool_pkl=rule_pool_dir / "initial_rule_pool.pkl",
        arms_json=arms_dir / "initial_arms.json",
        retrain_eval_summary=arm_run_dir / "retrain_eval" / "summary.json",
    )


def _ensure_empty_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise SystemExit(f"Directory not empty: {path}. Use --force to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _arm_run_has_iterations(arm_run_dir: Path) -> bool:
    if not arm_run_dir.exists():
        return False
    for child in arm_run_dir.iterdir():
        if child.is_dir() and child.name.startswith("iter_"):
            return True
    return False


def run_pipeline(
    *,
    run_dir: str | Path,
    model_dir: str | Path,
    target_relation: str,
    dataset_dir: str | Path,
    target_triples: str | Path,
    candidate_triples: str | Path,
    candidate_source: str = "local",
    web_llm_model: str = "gpt-4o",
    web_use_web_search: bool = True,
    web_max_targets_total_per_iteration: int = 20,
    web_max_triples_per_iteration: int = 200,
    disable_entity_linking: bool = False,
    # rule pool
    n_rules: int = 20,
    sort_by: str = "support",
    mode: str = "entire",
    min_head_coverage: float = 0.01,
    min_pca_conf: float = 0.01,
    lower_percentile: float = 90.0,
    k_neighbor: int = 1,
    # arms
    k_pairs: int = 20,
    pair_support_source: str = "candidate",
    max_witness_per_head: Optional[int] = None,
    rule_top_k: Optional[int] = None,
    rule_sort_by: str = "pca_conf",
    arms_min_pca_conf: Optional[float] = None,
    arms_min_head_coverage: Optional[float] = None,
    exclude_relation_pattern: Optional[Sequence[str]] = None,
    rule_filter_config: Optional[str | Path] = None,
    # arm-run
    n_iter: int = 1,
    k_sel: int = 3,
    n_targets_per_arm: int = 50,
    selector_strategy: str = "ucb",
    selector_exploration_param: float = 1.0,
    selector_epsilon: float = 0.1,
    witness_weight: float = 1.0,
    evidence_weight: float = 1.0,
    disable_incident_triples: bool = False,
    max_incident_candidate_triples_per_iteration: Optional[int] = None,
    relation_priors_path: Optional[str | Path] = None,
    disable_relation_priors: bool = False,
    compute_relation_priors: bool = False,
    relation_priors_out: Optional[str | Path] = None,
    xr_min_count_x7: int = 50,
    xr_max_samples_x3_per_relation: int = 2000,
    xr_max_samples_x7_per_relation: int = 5000,
    xr_weight_x2: float = 0.0,
    xr_weight_x3: float = 0.0,
    xr_weight_x4: float = 0.0,
    xr_weight_x7: float = 1.0,
    # eval
    after_mode: str = "retrain",
    embedding_config: str | Path = "./config_embeddings.json",
    num_epochs: int = 2,
    force_retrain: bool = False,
    model_before_dir: Optional[str | Path] = None,
    model_after_dir: Optional[str | Path] = None,
    exclude_predicate: Optional[Sequence[str]] = None,
    # runner behavior
    force: bool = False,
) -> RunnerPaths:
    """Run the full pipeline.

    This function is import-safe (no argparse) and is intended for unit testing
    and for reuse by other orchestration code.

    Returns:
        Resolved paths and key artifact locations.
    """

    paths = _resolve_paths(run_dir)

    # The KGE used for initial rule pool (model_dir) is also the natural default
    # for the "before" model in retrain/eval.
    if model_before_dir is None:
        model_before_dir = model_dir

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.rule_pool_dir.mkdir(parents=True, exist_ok=True)
    paths.arms_dir.mkdir(parents=True, exist_ok=True)
    paths.arm_run_dir.mkdir(parents=True, exist_ok=True)

    # Optional Step 0: compute relation priors from before KGE + train triples.
    # Precedence: explicit relation_priors_path > computed priors > auto-detect by arm pipeline.
    if relation_priors_path is None and compute_relation_priors:
        out_path = Path(relation_priors_out) if relation_priors_out else (paths.run_dir / "relation_priors" / "relation_priors.json")
        cfg_pr = RelationPriorConfig(
            max_samples_x3_per_relation=int(xr_max_samples_x3_per_relation),
            max_samples_x7_per_relation=int(xr_max_samples_x7_per_relation),
            random_seed=0,
            min_count_x7=int(xr_min_count_x7),
            weight_x2=float(xr_weight_x2),
            weight_x3=float(xr_weight_x3),
            weight_x4=float(xr_weight_x4),
            weight_x7=float(xr_weight_x7),
        )
        logger.info("[run] compute_relation_priors -> %s", out_path)
        relation_priors_path = compute_and_save_relation_priors(
            dataset_dir=str(dataset_dir),
            model_before_dir=str(model_dir),
            output_path=str(out_path),
            cfg=cfg_pr,
        )

    # Step 1: rule pool
    if paths.rule_pool_pkl.exists() and not force:
        logger.info("[skip] rule_pool exists: %s", paths.rule_pool_pkl)
    else:
        if force:
            _ensure_empty_dir(paths.rule_pool_dir, force=True)
        logger.info("[run] build_initial_rule_pool -> %s", paths.rule_pool_dir)
        build_initial_rule_pool(
            model_dir=str(model_dir),
            target_relation=target_relation,
            output_dir=str(paths.rule_pool_dir),
            n_rules=int(n_rules),
            sort_by=sort_by,
            mode=mode,
            min_head_coverage=float(min_head_coverage),
            min_pca_conf=float(min_pca_conf),
            lower_percentile=float(lower_percentile),
            k_neighbor=int(k_neighbor),
        )
        if not paths.rule_pool_pkl.exists():
            raise RuntimeError(f"Expected rule pool pickle not found: {paths.rule_pool_pkl}")

    # Step 2: arms
    if paths.arms_json.exists() and not force:
        logger.info("[skip] arms exist: %s", paths.arms_json)
    else:
        if force:
            _ensure_empty_dir(paths.arms_dir, force=True)
        logger.info("[run] build_initial_arms -> %s", paths.arms_dir)
        build_initial_arm_pool(
            rule_pool_path=str(paths.rule_pool_pkl),
            target_triples_path=str(target_triples),
            candidate_triples_path=str(candidate_triples),
            dir_triples=str(dataset_dir),
            include_train_removed=False,
            output_dir=str(paths.arms_dir),
            k_pairs=int(k_pairs),
            max_witness_per_head=max_witness_per_head,
            rule_top_k=rule_top_k,
            rule_sort_by=rule_sort_by,
            min_pca_conf=arms_min_pca_conf,
            min_head_coverage=arms_min_head_coverage,
            exclude_relation_patterns=list(exclude_relation_pattern) if exclude_relation_pattern else None,
            rule_filter_config_path=str(rule_filter_config) if rule_filter_config else None,
            pair_support_source=str(pair_support_source),
        )
        if not paths.arms_json.exists():
            raise RuntimeError(f"Expected arms JSON not found: {paths.arms_json}")

    # Step 3: arm refinement
    if _arm_run_has_iterations(paths.arm_run_dir) and not force:
        logger.info("[skip] arm_run iterations exist under: %s", paths.arm_run_dir)
    else:
        if force:
            _ensure_empty_dir(paths.arm_run_dir, force=True)
        logger.info("[run] arm refinement -> %s", paths.arm_run_dir)
        cfg = ArmPipelineConfig(
            base_output_path=str(paths.arm_run_dir),
            n_iter=int(n_iter),
            k_sel=int(k_sel),
            n_targets_per_arm=int(n_targets_per_arm),
            max_witness_per_head=max_witness_per_head,
            selector_strategy=selector_strategy,
            selector_exploration_param=float(selector_exploration_param),
            selector_epsilon=float(selector_epsilon),
            witness_weight=float(witness_weight),
            evidence_weight=float(evidence_weight),
            add_incident_candidate_triples_for_new_entities=not bool(disable_incident_triples),
            max_incident_candidate_triples_per_iteration=max_incident_candidate_triples_per_iteration,
            relation_priors_path=str(relation_priors_path) if relation_priors_path else None,
            disable_relation_priors=bool(disable_relation_priors),
            candidate_source=str(candidate_source),
            web_llm_model=str(web_llm_model),
            web_use_web_search=bool(web_use_web_search),
            web_max_targets_total_per_iteration=int(web_max_targets_total_per_iteration),
            web_max_triples_per_iteration=int(web_max_triples_per_iteration),
            web_enable_entity_linking=not bool(disable_entity_linking),
        )
        pipe = ArmDrivenKGRefinementPipeline.from_paths(
            config=cfg,
            initial_arms_path=str(paths.arms_json),
            rule_pool_pkl=str(paths.rule_pool_pkl),
            dir_triples=str(dataset_dir),
            target_triples_path=str(target_triples),
            candidate_triples_path=str(candidate_triples),
        )
        pipe.run()

    # Step 4: retrain + eval
    if paths.retrain_eval_summary.exists() and not force:
        logger.info("[skip] retrain_eval exists: %s", paths.retrain_eval_summary)
    else:
        logger.info("[run] retrain_and_evaluate -> %s", paths.arm_run_dir)
        retrain_and_evaluate(
            run_dir=paths.arm_run_dir,
            dataset_dir=dataset_dir,
            target_triples=target_triples,
            model_before_dir=model_before_dir,
            model_after_dir=model_after_dir,
            exclude_predicate=exclude_predicate,
            after_mode=after_mode,
            embedding_config=embedding_config,
            num_epochs=int(num_epochs),
            force_retrain=bool(force_retrain),
        )
        if not paths.retrain_eval_summary.exists():
            raise RuntimeError(f"Expected summary.json not found: {paths.retrain_eval_summary}")

    return paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full arm refinement pipeline")

    p.add_argument("--run_dir", required=True, help="Experiment run root (outputs are created under it)")
    p.add_argument("--model_dir", required=True, help="Trained KGE directory for initial rule pool")
    p.add_argument("--target_relation", required=True, help="Target relation (e.g., /people/person/nationality)")
    p.add_argument("--dataset_dir", required=True, help="Dataset directory containing train/valid/test")
    p.add_argument("--target_triples", required=True, help="Target triples TSV")
    p.add_argument("--candidate_triples", required=True, help="Candidate triples TSV (e.g., train_removed.txt)")

    p.add_argument(
        "--candidate_source",
        default="local",
        choices=["local", "web"],
        help="Candidate source for arm-run evidence acquisition: local (train_removed) or web (LLM/web search)",
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

    # rule pool
    p.add_argument("--n_rules", type=int, default=20)
    p.add_argument("--sort_by", default="support")
    p.add_argument("--mode", choices=["entire", "high-score"], default="entire")
    p.add_argument("--min_head_coverage", type=float, default=0.01)
    p.add_argument("--min_pca_conf", type=float, default=0.01)
    p.add_argument("--lower_percentile", type=float, default=90.0)
    p.add_argument("--k_neighbor", type=int, default=1)

    # arms
    p.add_argument("--k_pairs", type=int, default=20)
    p.add_argument(
        "--pair_support_source",
        choices=["candidate", "train"],
        default="candidate",
        help=(
            "Which triple set to use when computing pair-arm support/Jaccard in Step2: "
            "candidate (default; typically train_removed) or train (train.txt only)."
        ),
    )
    p.add_argument("--max_witness_per_head", type=int, default=None)
    p.add_argument("--rule_top_k", type=int, default=None)
    p.add_argument("--rule_sort_by", default="pca_conf")
    p.add_argument("--arms_min_pca_conf", type=float, default=None)
    p.add_argument("--arms_min_head_coverage", type=float, default=None)
    p.add_argument("--exclude_relation_pattern", action="append", default=None)
    p.add_argument("--rule_filter_config", default=None)

    # arm-run
    p.add_argument("--n_iter", type=int, default=1)
    p.add_argument("--k_sel", type=int, default=3)
    p.add_argument("--n_targets_per_arm", type=int, default=50)
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
            "By default, the arm-run may add extra candidate triples incident to new entities to avoid dangling entities."
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
            "If omitted, auto-detect <dataset_dir>/relation_priors.json when present."
        ),
    )
    p.add_argument(
        "--disable_relation_priors",
        action="store_true",
        help=(
            "Disable relation priors (X_r) entirely, including auto-detect of <dataset_dir>/relation_priors.json. "
            "Use this to enforce priors=off."
        ),
    )
    p.add_argument("--compute_relation_priors", action="store_true", help="Compute relation priors (X_r) before arm-run")
    p.add_argument(
        "--relation_priors_out",
        default=None,
        help="Output path for computed priors (default: <run_dir>/relation_priors/relation_priors.json)",
    )
    p.add_argument("--xr_min_count_x7", type=int, default=50)
    p.add_argument("--xr_max_samples_x3_per_relation", type=int, default=2000)
    p.add_argument("--xr_max_samples_x7_per_relation", type=int, default=5000)
    p.add_argument("--xr_weight_x2", type=float, default=0.0)
    p.add_argument("--xr_weight_x3", type=float, default=0.0)
    p.add_argument("--xr_weight_x4", type=float, default=0.0)
    p.add_argument("--xr_weight_x7", type=float, default=1.0)

    # eval
    p.add_argument("--after_mode", choices=["load", "retrain"], default="retrain")
    p.add_argument("--embedding_config", default="./config_embeddings.json")
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--model_before_dir", default=None)
    p.add_argument("--model_after_dir", default=None)
    p.add_argument("--exclude_predicate", action="append", default=None)

    # runner
    p.add_argument("--force", action="store_true", help="Overwrite existing step outputs under run_dir")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(
        run_dir=args.run_dir,
        model_dir=args.model_dir,
        target_relation=args.target_relation,
        dataset_dir=args.dataset_dir,
        target_triples=args.target_triples,
        candidate_triples=args.candidate_triples,
        candidate_source=args.candidate_source,
        web_llm_model=args.web_llm_model,
        web_use_web_search=not bool(args.disable_web_search),
        web_max_targets_total_per_iteration=int(args.web_max_targets_total_per_iteration),
        web_max_triples_per_iteration=int(args.web_max_triples_per_iteration),
        disable_entity_linking=bool(args.disable_entity_linking),
        n_rules=args.n_rules,
        sort_by=args.sort_by,
        mode=args.mode,
        min_head_coverage=args.min_head_coverage,
        min_pca_conf=args.min_pca_conf,
        lower_percentile=args.lower_percentile,
        k_neighbor=args.k_neighbor,
        k_pairs=args.k_pairs,
        pair_support_source=args.pair_support_source,
        max_witness_per_head=args.max_witness_per_head,
        rule_top_k=args.rule_top_k,
        rule_sort_by=args.rule_sort_by,
        arms_min_pca_conf=args.arms_min_pca_conf,
        arms_min_head_coverage=args.arms_min_head_coverage,
        exclude_relation_pattern=args.exclude_relation_pattern,
        rule_filter_config=args.rule_filter_config,
        n_iter=args.n_iter,
        k_sel=args.k_sel,
        n_targets_per_arm=args.n_targets_per_arm,
        selector_strategy=args.selector_strategy,
        selector_exploration_param=args.selector_exploration_param,
        selector_epsilon=args.selector_epsilon,
        witness_weight=args.witness_weight,
        evidence_weight=args.evidence_weight,
        relation_priors_path=args.relation_priors_path,
        disable_relation_priors=bool(args.disable_relation_priors),
        compute_relation_priors=bool(args.compute_relation_priors),
        relation_priors_out=args.relation_priors_out,
        xr_min_count_x7=int(args.xr_min_count_x7),
        xr_max_samples_x3_per_relation=int(args.xr_max_samples_x3_per_relation),
        xr_max_samples_x7_per_relation=int(args.xr_max_samples_x7_per_relation),
        xr_weight_x2=float(args.xr_weight_x2),
        xr_weight_x3=float(args.xr_weight_x3),
        xr_weight_x4=float(args.xr_weight_x4),
        xr_weight_x7=float(args.xr_weight_x7),
        after_mode=args.after_mode,
        embedding_config=args.embedding_config,
        num_epochs=args.num_epochs,
        force_retrain=args.force_retrain,
        model_before_dir=args.model_before_dir,
        model_after_dir=args.model_after_dir,
        exclude_predicate=args.exclude_predicate,
        force=args.force,
    )


if __name__ == "__main__":
    main()
