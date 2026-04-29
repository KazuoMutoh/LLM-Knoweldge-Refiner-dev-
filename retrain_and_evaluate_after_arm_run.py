"""Evaluate KGE before/after an arm-driven refinement run.

This script aggregates `accepted_evidence_triples.tsv` from an arm-run output
folder, creates an updated dataset (train ∪ evidence), and compares KGE metrics
and target triple scores.

Modes:
    - before: always loaded from an existing model directory.
    - after: either loaded from an existing model directory, or retrained on
      the updated triples directory and saved to the after model directory.

Default policy: include all accepted evidence triples as-is.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from simple_active_refine.dataset_update import (
    EvidenceAggregationResult,
    aggregate_accepted_added_triples,
    aggregate_accepted_evidence_triples,
    create_updated_triples_dir,
)
from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.evaluation import IterationEvaluator, IterationMetrics
from simple_active_refine.io_utils import read_triples, write_triples, save_json
from simple_active_refine.kgfit_cache_update import ensure_kgfit_cache_complete
from simple_active_refine.util import get_logger

logger = get_logger(__name__)

Triple = Tuple[str, str, str]


def _copy_split_files_to_model_dir(updated_triples_dir: Path, model_dir: Path) -> None:
    """Copy test/valid splits into model directory for evaluate().

    KnowledgeGraphEmbedding.evaluate() looks for test.txt (and falls back to
    valid.txt) under kge.dir_save. PyKEEN's save_to_directory does not include
    dataset split text files, so we copy them from updated_triples_dir.
    """

    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("test.txt", "valid.txt"):
        src = updated_triples_dir / name
        if not src.exists():
            continue
        dst = model_dir / name
        shutil.copy2(src, dst)


def _ensure_model_dir_has_splits(*, source_triples_dir: Path, model_dir: Path) -> None:
    """Ensure model_dir has test/valid split files for evaluation."""

    for name in ("test.txt", "valid.txt"):
        dst = model_dir / name
        if dst.exists():
            continue
        src = source_triples_dir / name
        if not src.exists():
            continue
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def _validate_trained_model_dir(model_dir: Path, label: str) -> None:
    """Validate that a directory looks like a saved PyKEEN run.

    The wrapper `KnowledgeGraphEmbedding` expects:
      - trained_model.pkl
      - training_triples/ (binary TriplesFactory)

    Args:
        model_dir: Directory to validate.
        label: Human-friendly label used in error messages.
    """

    if not model_dir.exists() or not model_dir.is_dir():
        logger.warning("%s model directory does not exist: %s", label, model_dir)
        raise SystemExit(2)
    if not (model_dir / "trained_model.pkl").exists():
        logger.warning("%s model is missing trained_model.pkl: %s", label, model_dir)
        raise SystemExit(2)
    if not (model_dir / "training_triples").exists():
        logger.warning("%s model is missing training_triples/: %s", label, model_dir)
        raise SystemExit(2)


def _load_embedding_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"embedding_config must be a JSON object, got: {type(cfg)}")
    return cfg


def _maybe_copy_cache_dir(*, source_triples_dir: Path, updated_triples_dir: Path, embedding_backend: str) -> None:
    """Copy required cache artifacts into updated triples directory when needed.

    KG-FIT training expects precomputed artifacts under:
      <dir_triples>/.cache/kgfit/

    When we create an updated triples directory, it does not include these
    artifacts by default. This helper copies them from the original dataset
    directory when available.
    """

    if embedding_backend != "kgfit":
        return

    src_cache = source_triples_dir / ".cache" / "kgfit"
    if not src_cache.exists():
        logger.warning("KG-FIT backend selected but cache directory not found: %s", src_cache)
        logger.warning("Expected artifacts under <dataset_dir>/.cache/kgfit/. Training may fail.")
        return

    dst_cache = updated_triples_dir / ".cache" / "kgfit"
    if dst_cache.exists() and any(dst_cache.iterdir()):
        return

    dst_cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_cache, dst_cache, dirs_exist_ok=True)
    logger.info("Copied KG-FIT cache artifacts: %s -> %s", src_cache, dst_cache)


def _train_after_kge(
    *,
    updated_triples_dir: Path,
    model_after_dir: Path,
    embedding_config_path: Path,
    num_epochs: int,
    force_retrain: bool,
) -> KnowledgeGraphEmbedding:
    """Train the 'after' KGE on updated triples and save it."""

    if model_after_dir.exists() and any(model_after_dir.iterdir()):
        if not force_retrain:
            logger.warning("after model directory is not empty: %s", model_after_dir)
            logger.warning("Use --force_retrain to overwrite.")
            raise SystemExit(2)
        shutil.rmtree(model_after_dir)

    model_after_dir.mkdir(parents=True, exist_ok=True)
    _copy_split_files_to_model_dir(updated_triples_dir, model_after_dir)

    cfg = _load_embedding_config(embedding_config_path)
    embedding_backend = cfg.pop("embedding_backend", "pykeen")
    kgfit_config = cfg.pop("kgfit", None)
    # Some configs may include top-level num_epochs, but this repo's PyKEEN
    # pipeline wrapper does not accept it as a keyword argument.
    cfg.pop("num_epochs", None)
    cfg["dir_triples"] = str(updated_triples_dir)
    cfg["dir_save"] = str(model_after_dir)

    training_kwargs = cfg.get("training_kwargs")
    if training_kwargs is None or not isinstance(training_kwargs, dict):
        training_kwargs = {}
        cfg["training_kwargs"] = training_kwargs
    training_kwargs["num_epochs"] = int(num_epochs)

    # For KG-FIT retraining on an updated dataset, we must use *local* cache
    # artifacts that include any new entities (e.g., web:*). Many configs point
    # to global dataset caches (e.g., /app/data/FB15k-237/.cache/kgfit), which
    # cannot cover newly introduced entities.
    if str(embedding_backend) == "kgfit":
        if kgfit_config is None or not isinstance(kgfit_config, dict):
            kgfit_config = {}

        local_cache = updated_triples_dir / ".cache" / "kgfit"
        local_cache.mkdir(parents=True, exist_ok=True)

        kgfit_config = dict(kgfit_config)
        kgfit_config["paths"] = {
            "name_embeddings": str(local_cache / "entity_name_embeddings.npy"),
            "desc_embeddings": str(local_cache / "entity_desc_embeddings.npy"),
            "meta": str(local_cache / "entity_embedding_meta.json"),
        }
        hierarchy_cfg = kgfit_config.get("hierarchy")
        if not isinstance(hierarchy_cfg, dict):
            hierarchy_cfg = {}
        hierarchy_cfg = dict(hierarchy_cfg)
        hierarchy_cfg.update(
            {
                "hierarchy_seed": str(local_cache / "hierarchy_seed.json"),
                "cluster_embeddings": str(local_cache / "cluster_embeddings.npy"),
                "neighbor_clusters": str(local_cache / "neighbor_clusters.json"),
            }
        )
        kgfit_config["hierarchy"] = hierarchy_cfg

        # Ensure local KG-FIT cache covers all entities used by updated_triples.
        ensure_kgfit_cache_complete(
            dir_triples=updated_triples_dir,
            cache_dir=local_cache,
            reshape_strategy=str(kgfit_config.get("reshape_strategy", "full")),
            embedding_dim=kgfit_config.get("embedding_dim"),
        )

    logger.info("Training after KGE: epochs=%d dir_triples=%s dir_save=%s", num_epochs, updated_triples_dir, model_after_dir)
    return KnowledgeGraphEmbedding.train_model(
        embedding_backend=embedding_backend,
        kgfit_config=kgfit_config,
        # Avoid running full rank-based evaluation inside PyKEEN's training
        # pipeline. This evaluation is memory-intensive (especially for
        # KG-FIT + high-dimensional embeddings) and is not needed for this
        # script because we run evaluation explicitly afterwards via
        # IterationEvaluator / KnowledgeGraphEmbedding.evaluate().
        skip_evaluation=True,
        **cfg,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain and evaluate KGE after arm-run")
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Arm-run output directory containing iter_*/accepted_evidence_triples.tsv",
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Original dataset directory containing train/valid/test",
    )
    p.add_argument(
        "--target_triples",
        type=str,
        required=True,
        help="Target triples TSV for score comparison (e.g., target_triples.txt)",
    )
    p.add_argument(
        "--model_before_dir",
        type=str,
        default=None,
        help="Directory containing pre-trained 'before' model (default: <run_dir>/retrain_eval/model_before)",
    )
    p.add_argument(
        "--model_after_dir",
        type=str,
        default=None,
        help="Directory containing pre-trained 'after' model (default: <run_dir>/retrain_eval/model_after)",
    )
    p.add_argument(
        "--exclude_predicate",
        type=str,
        action="append",
        default=None,
        help="Optional predicate to exclude from added evidence (repeatable). Default: include all.",
    )

    p.add_argument(
        "--after_mode",
        type=str,
        default="retrain",
        choices=["load", "retrain"],
        help="How to obtain the 'after' model: load an existing model dir, or retrain on updated triples.",
    )
    p.add_argument(
        "--embedding_config",
        type=str,
        default="./config_embeddings.json",
        help="Embedding config JSON used when --after_mode retrain.",
    )
    p.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of epochs for after-model retraining when --after_mode retrain.",
    )
    p.add_argument(
        "--force_retrain",
        action="store_true",
        help="Allow overwriting a non-empty after model dir when --after_mode retrain.",
    )
    return p.parse_args()


def run(
    *,
    run_dir: str | Path,
    dataset_dir: str | Path,
    target_triples: str | Path,
    model_before_dir: str | Path | None = None,
    model_after_dir: str | Path | None = None,
    exclude_predicate: Optional[Sequence[str]] = None,
    after_mode: str = "retrain",
    embedding_config: str | Path = "./config_embeddings.json",
    num_epochs: int = 2,
    force_retrain: bool = False,
) -> Path:
    """Aggregate evidence, build updated triples, retrain/load after model, then evaluate.

    Notes:
        This function evaluates the before/after models sequentially to keep
        peak memory low in CPU-only environments.
    """

    import gc

    run_dir = Path(run_dir)
    dataset_dir = Path(dataset_dir)
    target_triples_path = Path(target_triples)
    embedding_config_path = Path(embedding_config)

    out_base = run_dir / "retrain_eval"
    updated_triples_dir = out_base / "updated_data"
    eval_dir = out_base / "evaluations"
    out_base.mkdir(parents=True, exist_ok=True)

    model_before_path = Path(model_before_dir) if model_before_dir is not None else (out_base / "model_before")
    model_after_path = Path(model_after_dir) if model_after_dir is not None else (out_base / "model_after")

    # 1) Aggregate evidence triples from the run
    agg = aggregate_accepted_added_triples(run_dir)
    added_triples = agg.evidence_triples

    # 2) Create updated triples directory: train ∪ evidence
    updated_result = create_updated_triples_dir(
        dataset_dir=dataset_dir,
        out_dir=updated_triples_dir,
        evidence_triples=added_triples,
        exclude_predicates=exclude_predicate,
    )
    updated_triples_dir = updated_result.out_dir

    # If we retrain with KG-FIT, we need cache artifacts under
    # updated_triples_dir/.cache/kgfit/.
    embedding_cfg = _load_embedding_config(embedding_config_path)
    embedding_backend = str(embedding_cfg.get("embedding_backend", "pykeen"))
    _maybe_copy_cache_dir(
        source_triples_dir=dataset_dir,
        updated_triples_dir=updated_triples_dir,
        embedding_backend=embedding_backend,
    )

    # 3) Validate/load before model
    _validate_trained_model_dir(model_before_path, label="before")
    _ensure_model_dir_has_splits(source_triples_dir=dataset_dir, model_dir=model_before_path)

    # 4) Evaluate sequentially
    target_triples_list: List[Triple] = read_triples(target_triples_path)

    def _mean(scores: List[float]) -> float:
        if not scores:
            return 0.0
        return float(sum(float(s) for s in scores) / len(scores))

    def _evaluate_one(
        *,
        label: str,
        kge: KnowledgeGraphEmbedding,
        target_triples_local: List[Triple],
    ) -> tuple[int, float, dict[str, float]]:
        n_triples = int(kge.triples.mapped_triples.shape[0])
        logger.info("  Calculating scores for %d target triples (%s)", len(target_triples_local), label)
        target_score = _mean(list(kge.score_triples(target_triples_local)))

        logger.info("  Evaluating knowledge graph embedding (%s)", label)
        eval_metrics = kge.evaluate(filtered=True, batch_size=1, slice_size=64)
        return n_triples, target_score, eval_metrics

    evaluator = IterationEvaluator()

    # before
    logger.info("Evaluating before model (sequential mode)")
    kge_before = KnowledgeGraphEmbedding(str(model_before_path), fit_normalization=False)
    n_triples_before, target_score_before, eval_before = _evaluate_one(
        label="before",
        kge=kge_before,
        target_triples_local=target_triples_list,
    )
    del kge_before
    gc.collect()

    # after
    logger.info("Preparing after model (sequential mode): after_mode=%s", after_mode)
    if after_mode == "load":
        _validate_trained_model_dir(model_after_path, label="after")
        kge_after = KnowledgeGraphEmbedding(str(model_after_path), fit_normalization=False)
    elif after_mode == "retrain":
        kge_after = _train_after_kge(
            updated_triples_dir=updated_triples_dir,
            model_after_dir=model_after_path,
            embedding_config_path=embedding_config_path,
            num_epochs=num_epochs,
            force_retrain=force_retrain,
        )
    else:
        raise ValueError(f"Unknown after_mode: {after_mode}")

    _ensure_model_dir_has_splits(source_triples_dir=dataset_dir, model_dir=model_after_path)

    n_triples_after, target_score_after, eval_after = _evaluate_one(
        label="after",
        kge=kge_after,
        target_triples_local=target_triples_list,
    )
    del kge_after
    gc.collect()

    target_score_change = target_score_after - target_score_before

    metrics = IterationMetrics(
        iteration=1,
        n_triples_before=n_triples_before,
        n_triples_after=n_triples_after,
        n_triples_added=updated_result.n_added_used,
        target_score_before=target_score_before,
        target_score_after=target_score_after,
        target_score_change=target_score_change,
        hits_at_1_before=float(eval_before.get("hits_at_1", 0.0)),
        hits_at_3_before=float(eval_before.get("hits_at_3", 0.0)),
        hits_at_10_before=float(eval_before.get("hits_at_10", 0.0)),
        mrr_before=float(eval_before.get("mean_reciprocal_rank", 0.0)),
        hits_at_1_after=float(eval_after.get("hits_at_1", 0.0)),
        hits_at_3_after=float(eval_after.get("hits_at_3", 0.0)),
        hits_at_10_after=float(eval_after.get("hits_at_10", 0.0)),
        mrr_after=float(eval_after.get("mean_reciprocal_rank", 0.0)),
        hits_at_1_change=float(eval_after.get("hits_at_1", 0.0)) - float(eval_before.get("hits_at_1", 0.0)),
        hits_at_3_change=float(eval_after.get("hits_at_3", 0.0)) - float(eval_before.get("hits_at_3", 0.0)),
        hits_at_10_change=float(eval_after.get("hits_at_10", 0.0)) - float(eval_before.get("hits_at_10", 0.0)),
        mrr_change=float(eval_after.get("mean_reciprocal_rank", 0.0)) - float(eval_before.get("mean_reciprocal_rank", 0.0)),
    )

    evaluator.metrics_history.append(metrics)
    evaluator._save_iteration_report(metrics, str(eval_dir))

    summary = {
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "target_triples": str(target_triples_path),
        "model_before_dir": str(model_before_path),
        "model_after_dir": str(model_after_path),
        "after_mode": after_mode,
        "embedding_config": str(embedding_config_path),
        "num_epochs": int(num_epochs),
        "force_retrain": bool(force_retrain),
        "aggregation": {
            "n_iterations_seen": agg.n_iterations_seen,
            "n_files_seen": agg.n_files_seen,
            "n_triples_total": agg.n_triples_total,
            "n_triples_unique": len(agg.evidence_triples),
        },
        "updated_dataset": {
            "n_train_before": updated_result.n_train_before,
            "n_added_used": updated_result.n_added_used,
            "n_train_after": updated_result.n_train_after,
            "excluded_predicates": sorted(updated_result.excluded_predicates),
        },
        "metrics": metrics.to_dict(),
    }
    save_json(out_base / "summary.json", summary)

    logger.info("Done. Outputs written to %s", out_base)
    return out_base


def main() -> None:
    args = _parse_args()
    run(
        run_dir=args.run_dir,
        dataset_dir=args.dataset_dir,
        target_triples=args.target_triples,
        model_before_dir=args.model_before_dir,
        model_after_dir=args.model_after_dir,
        exclude_predicate=args.exclude_predicate,
        after_mode=args.after_mode,
        embedding_config=args.embedding_config,
        num_epochs=args.num_epochs,
        force_retrain=args.force_retrain,
    )


if __name__ == "__main__":
    main()
