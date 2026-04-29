"""Run random-add baseline experiments (sequential seeds).

This runner creates arm-run compatible directories:
  <run_dir>/arm_run/iter_1/accepted_added_triples.tsv

Then it calls retrain_and_evaluate_after_arm_run.py to retrain/evaluate.

Rationale:
- We want a fair baseline vs UCB by adding the same number of triples sampled
  from train_removed.txt (without replacement).
- Running sequentially avoids GPU/CPU resource contention.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run random baseline batch.")

    parser.add_argument(
        "--base_out_dir",
        type=Path,
        default=Path("/app/experiments/20260123"),
        help="Base output directory (date folder).",
    )
    parser.add_argument(
        "--run_suffix",
        type=str,
        default="20260123a",
        help="Suffix used in run_dir names.",
    )

    parser.add_argument(
        "--candidate_triples",
        type=Path,
        default=Path("/app/experiments/test_data_for_nationality_v3/train_removed.txt"),
        help="Path to train_removed.txt (3-col TSV).",
    )
    parser.add_argument(
        "--n_add",
        type=int,
        default=10232,
        help="Number of triples to add (must match UCB n_added_used).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds to run sequentially.",
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/app/experiments/test_data_for_nationality_v3_kgfit"),
        help="dir_triples used for retraining (KG-FIT dataset).",
    )
    parser.add_argument(
        "--target_triples",
        type=Path,
        default=Path("/app/experiments/test_data_for_nationality_v3/target_triples.txt"),
        help="Target triples TSV.",
    )
    parser.add_argument(
        "--model_before_dir",
        type=Path,
        default=Path("/app/models/20260122/fb15k237_kgfit_pairre_nationality_v3_before_ep100"),
        help="Before model directory.",
    )
    parser.add_argument(
        "--embedding_config",
        type=Path,
        default=Path("/app/config_embeddings_kgfit_pairre_fb15k237.json"),
        help="Embedding config JSON.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs for retraining.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only print planned commands without executing.",
    )
    return parser


def _run_one_seed(
    *,
    seed: int,
    base_out_dir: Path,
    run_suffix: str,
    n_add: int,
    candidate_triples: Path,
    dataset_dir: Path,
    target_triples: Path,
    model_before_dir: Path,
    embedding_config: Path,
    num_epochs: int,
    dry_run: bool,
) -> None:
    run_dir = (
        base_out_dir
        / f"exp_random_add{n_add}_priors_off_kgfit_pairre_ep{num_epochs}_seed{seed}_{run_suffix}"
    )
    arm_run_dir = run_dir / "arm_run"
    iter1_dir = arm_run_dir / "iter_1"
    accepted_tsv = iter1_dir / "accepted_added_triples.tsv"

    run_dir.mkdir(parents=True, exist_ok=True)
    iter1_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "seed": seed,
        "n_add": n_add,
        "candidate_triples": str(candidate_triples),
        "dataset_dir": str(dataset_dir),
        "target_triples": str(target_triples),
        "model_before_dir": str(model_before_dir),
        "embedding_config": str(embedding_config),
        "num_epochs": num_epochs,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    (run_dir / "random_add_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    sample_cmd = [
        sys.executable,
        "/app/scripts/sample_random_triples.py",
        "--input_tsv",
        str(candidate_triples),
        "--output_tsv",
        str(accepted_tsv),
        "--n",
        str(n_add),
        "--seed",
        str(seed),
    ]

    retrain_cmd = [
        sys.executable,
        "/app/retrain_and_evaluate_after_arm_run.py",
        "--run_dir",
        str(arm_run_dir),
        "--dataset_dir",
        str(dataset_dir),
        "--target_triples",
        str(target_triples),
        "--model_before_dir",
        str(model_before_dir),
        "--after_mode",
        "retrain",
        "--embedding_config",
        str(embedding_config),
        "--num_epochs",
        str(num_epochs),
    ]

    print("=" * 80)
    print(f"[seed={seed}] run_dir={run_dir}")
    print("[sample]", " ".join(sample_cmd))
    print("[retrain]", " ".join(retrain_cmd))

    if dry_run:
        return

    subprocess.run(sample_cmd, check=True)

    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as logf:
        subprocess.run(retrain_cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    args.base_out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        _run_one_seed(
            seed=seed,
            base_out_dir=args.base_out_dir,
            run_suffix=args.run_suffix,
            n_add=args.n_add,
            candidate_triples=args.candidate_triples,
            dataset_dir=args.dataset_dir,
            target_triples=args.target_triples,
            model_before_dir=args.model_before_dir,
            embedding_config=args.embedding_config,
            num_epochs=args.num_epochs,
            dry_run=args.dry_run,
        )

    print("Batch completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
