"""Run UCB then matched-N Random(seed=0) and generate a comparison report.

This script automates the experiment described in:
- docs/records/REC-20260123-UCB_VS_RANDOM_3REL_KGFIT_PAIRRE-001.md

Workflow (per relation):
  1) Run UCB via run_full_arm_pipeline.py with priors=off and incident=off
  2) Read N = summary.json.updated_dataset.n_added_used
  3) Sample N triples from train_removed.txt with seed=0 into an arm-run compatible path
  4) Retrain/evaluate after Random add using retrain_and_evaluate_after_arm_run.py
  5) Generate a markdown report including:
      - summary.json metrics comparison
      - minmax(train) normalized target rescoring (intersection of known targets)

Notes:
- This script is intentionally sequential (GPU heavy).
- Random sampling is without replacement.

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Allow running without setting PYTHONPATH
sys.path.insert(0, "/app")

from simple_active_refine.embedding import KnowledgeGraphEmbedding


Triple = Tuple[str, str, str]


def _run(cmd: Sequence[str], *, log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(list(cmd), check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(list(cmd), check=True, stdout=f, stderr=subprocess.STDOUT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_target(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split("\t") if "\t" in s else s.split()
        if len(parts) < 3:
            continue
        triples.append((parts[0], parts[1], parts[2]))
    return triples


def _is_known(kge: KnowledgeGraphEmbedding, triple: Triple) -> bool:
    h, r, t = triple
    return (
        (h in kge.triples.entity_to_id)
        and (t in kge.triples.entity_to_id)
        and (r in kge.triples.relation_to_id)
    )


def _delta_stats(before: Sequence[float], after: Sequence[float]) -> Dict[str, float | int]:
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    d = a - b
    return {
        "n": int(d.size),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p25": float(np.percentile(d, 25)),
        "p75": float(np.percentile(d, 75)),
        "improved_frac": float(np.mean(d > 0.0)),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
    }


def _fmt(x: float | int) -> str:
    if isinstance(x, int):
        return str(x)
    return f"{x:.6g}"


def _summary_metrics_row(summary: dict) -> Dict[str, float | int | str]:
    m = summary.get("metrics", {})
    u = summary.get("updated_dataset", {})
    return {
        "n_added_used": int(u.get("n_added_used", -1)),
        "target_score_change": float(m.get("target_score_change", float("nan"))),
        "mrr_change": float(m.get("mrr_change", float("nan"))),
        "hits@1_change": float(m.get("hits_at_1_change", float("nan"))),
        "hits@3_change": float(m.get("hits_at_3_change", float("nan"))),
        "hits@10_change": float(m.get("hits_at_10_change", float("nan"))),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run UCB vs matched-N Random(seed=0) and generate a report."
    )

    p.add_argument("--base_out_dir", type=Path, required=True)
    p.add_argument("--run_suffix", type=str, required=True)

    p.add_argument("--relation_name", type=str, required=True)
    p.add_argument("--target_relation", type=str, required=True)

    p.add_argument("--dataset_dir", type=Path, required=True)
    p.add_argument("--target_triples", type=Path, required=True)
    p.add_argument("--candidate_triples", type=Path, required=True)

    p.add_argument("--model_before_dir", type=Path, required=True)
    p.add_argument("--embedding_config", type=Path, required=True)
    p.add_argument("--num_epochs", type=int, default=100)

    # Arm/pipeline settings (default: match REC-20260123-UCB_VS_RANDOM_3REL...)
    p.add_argument("--n_rules", type=int, default=10)
    p.add_argument("--k_pairs", type=int, default=0)
    p.add_argument("--n_iter", type=int, default=25)
    p.add_argument("--k_sel", type=int, default=3)
    p.add_argument("--n_targets_per_arm", type=int, default=20)

    p.add_argument("--force", action="store_true", help="Overwrite existing run dirs")
    p.add_argument("--dry_run", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.base_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) UCB run
    ucb_run_dir = (
        args.base_out_dir
        / f"exp_ucb_arm{args.n_rules}_priors_off_kgfit_pairre_ep{args.num_epochs}_{args.relation_name}_head_incident_v1_{args.run_suffix}"
    )
    ucb_log = ucb_run_dir / "run.log"

    ucb_cmd = [
        sys.executable,
        "/app/run_full_arm_pipeline.py",
        "--run_dir",
        str(ucb_run_dir),
        "--model_dir",
        str(args.model_before_dir),
        "--target_relation",
        args.target_relation,
        "--dataset_dir",
        str(args.dataset_dir),
        "--target_triples",
        str(args.target_triples),
        "--candidate_triples",
        str(args.candidate_triples),
        "--n_rules",
        str(args.n_rules),
        "--k_pairs",
        str(args.k_pairs),
        "--n_iter",
        str(args.n_iter),
        "--k_sel",
        str(args.k_sel),
        "--n_targets_per_arm",
        str(args.n_targets_per_arm),
        "--selector_strategy",
        "ucb",
        "--disable_relation_priors",
        "--disable_incident_triples",
        "--after_mode",
        "retrain",
        "--embedding_config",
        str(args.embedding_config),
        "--num_epochs",
        str(args.num_epochs),
    ]
    if args.force:
        ucb_cmd.append("--force")

    ucb_summary_path = ucb_run_dir / "arm_run" / "retrain_eval" / "summary.json"

    print("=" * 80)
    print("[UCB] run_dir=", ucb_run_dir)
    print("[UCB] cmd=", " ".join(ucb_cmd))
    if not args.dry_run:
        _run(ucb_cmd, log_path=ucb_log)

    # 2) Determine N
    if args.dry_run:
        print("[dry_run] skipping N extraction")
        return 0

    if not ucb_summary_path.exists():
        raise SystemExit(f"UCB summary.json not found: {ucb_summary_path}")

    ucb_summary = _read_json(ucb_summary_path)
    n_add = int(ucb_summary.get("updated_dataset", {}).get("n_added_used", -1))
    if n_add <= 0:
        raise SystemExit(f"Invalid n_added_used in UCB summary: {n_add}")

    # 3) Random(seed=0) run
    rand_run_dir = (
        args.base_out_dir
        / f"exp_random_add{n_add}_priors_off_kgfit_pairre_ep{args.num_epochs}_{args.relation_name}_head_incident_v1_seed0_{args.run_suffix}"
    )
    rand_arm_run_dir = rand_run_dir / "arm_run"
    rand_iter1_dir = rand_arm_run_dir / "iter_1"
    rand_log = rand_run_dir / "run.log"
    rand_iter1_dir.mkdir(parents=True, exist_ok=True)

    sample_cmd = [
        sys.executable,
        "/app/scripts/sample_random_triples.py",
        "--input_tsv",
        str(args.candidate_triples),
        "--output_tsv",
        str(rand_iter1_dir / "accepted_added_triples.tsv"),
        "--n",
        str(n_add),
        "--seed",
        "0",
    ]
    retrain_cmd = [
        sys.executable,
        "/app/retrain_and_evaluate_after_arm_run.py",
        "--run_dir",
        str(rand_arm_run_dir),
        "--dataset_dir",
        str(args.dataset_dir),
        "--target_triples",
        str(args.target_triples),
        "--model_before_dir",
        str(args.model_before_dir),
        "--after_mode",
        "retrain",
        "--embedding_config",
        str(args.embedding_config),
        "--num_epochs",
        str(args.num_epochs),
    ]

    print("=" * 80)
    print(f"[Random(seed=0)] n_add={n_add} run_dir={rand_run_dir}")
    print("[Random] sample=", " ".join(sample_cmd))
    print("[Random] retrain=", " ".join(retrain_cmd))

    if args.force and rand_run_dir.exists():
        # Keep it simple: require user to clean manually or re-run with a new suffix.
        # (Avoid deleting large directories accidentally.)
        print(f"[warn] run_dir exists; --force does not delete for Random: {rand_run_dir}")

    _run(sample_cmd)
    _run(retrain_cmd, log_path=rand_log)

    rand_summary_path = rand_arm_run_dir / "retrain_eval" / "summary.json"
    if not rand_summary_path.exists():
        raise SystemExit(f"Random summary.json not found: {rand_summary_path}")
    rand_summary = _read_json(rand_summary_path)

    # 4) Minmax(train) target rescoring comparison
    target_triples = _read_target(args.target_triples)
    if not target_triples:
        raise SystemExit(f"No target triples found: {args.target_triples}")

    kge_before = KnowledgeGraphEmbedding(model_dir=str(args.model_before_dir), fit_normalization=True)
    ucb_after_dir = Path(ucb_summary["model_after_dir"])
    rand_after_dir = Path(rand_summary["model_after_dir"])

    kge_ucb = KnowledgeGraphEmbedding(model_dir=str(ucb_after_dir), fit_normalization=True)
    kge_rand = KnowledgeGraphEmbedding(model_dir=str(rand_after_dir), fit_normalization=True)

    kept: List[Triple] = []
    skipped: List[Triple] = []
    for tr in target_triples:
        if _is_known(kge_before, tr) and _is_known(kge_ucb, tr) and _is_known(kge_rand, tr):
            kept.append(tr)
        else:
            skipped.append(tr)

    before_scores = list(kge_before.score_triples(kept, normalize=True, norm_method="minmax"))
    ucb_scores = list(kge_ucb.score_triples(kept, normalize=True, norm_method="minmax"))
    rand_scores = list(kge_rand.score_triples(kept, normalize=True, norm_method="minmax"))

    stats_ucb = _delta_stats(before_scores, ucb_scores)
    stats_rand = _delta_stats(before_scores, rand_scores)

    # 5) Report
    report_path = (
        args.base_out_dir
        / f"compare_ucb_vs_random_seed0_{args.relation_name}_head_incident_v1_{args.run_suffix}.md"
    )

    ucb_row = _summary_metrics_row(ucb_summary)
    rand_row = _summary_metrics_row(rand_summary)

    md: List[str] = []
    md.append(f"# UCB vs Random(seed=0): {args.relation_name} (head_incident_v1)")
    md.append("")
    md.append("## Runs")
    md.append("")
    md.append(f"- UCB run_dir: `{ucb_run_dir}`")
    md.append(f"- Random run_dir: `{rand_run_dir}`")
    md.append("")

    md.append("## Summary.json metrics")
    md.append("")
    md.append("| condition | n_added_used | target_score_change | mrr_change | hits@1_change | hits@3_change | hits@10_change |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    md.append(
        "| UCB | "
        + " | ".join(
            [
                _fmt(int(ucb_row["n_added_used"])),
                _fmt(float(ucb_row["target_score_change"])),
                _fmt(float(ucb_row["mrr_change"])),
                _fmt(float(ucb_row["hits@1_change"])),
                _fmt(float(ucb_row["hits@3_change"])),
                _fmt(float(ucb_row["hits@10_change"])),
            ]
        )
        + " |"
    )
    md.append(
        "| Random(seed=0) | "
        + " | ".join(
            [
                _fmt(int(rand_row["n_added_used"])),
                _fmt(float(rand_row["target_score_change"])),
                _fmt(float(rand_row["mrr_change"])),
                _fmt(float(rand_row["hits@1_change"])),
                _fmt(float(rand_row["hits@3_change"])),
                _fmt(float(rand_row["hits@10_change"])),
            ]
        )
        + " |"
    )
    md.append("")

    md.append("## Minmax(train) normalized target rescoring")
    md.append("")
    md.append(
        "Target triples are re-scored with the fixed before model and each after model, "
        "using per-model minmax(train) normalization; Δ is computed on the intersection of target triples "
        "known to all models in this comparison."
    )
    md.append("")
    md.append(f"- target_triples: `{args.target_triples}`")
    md.append(f"- before model: `{args.model_before_dir}`")
    md.append(f"- after model (UCB): `{ucb_after_dir}`")
    md.append(f"- after model (Random): `{rand_after_dir}`")
    md.append(
        f"- usable targets (intersection): {len(kept)} (skipped {len(skipped)} / total {len(target_triples)})"
    )
    md.append("")
    md.append("| condition | Δmean | Δmedian | improved_frac | p25 | p75 | min | max | n |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    md.append(
        "| UCB | "
        + " | ".join(
            [
                _fmt(stats_ucb["mean"]),
                _fmt(stats_ucb["median"]),
                _fmt(stats_ucb["improved_frac"]),
                _fmt(stats_ucb["p25"]),
                _fmt(stats_ucb["p75"]),
                _fmt(stats_ucb["min"]),
                _fmt(stats_ucb["max"]),
                _fmt(stats_ucb["n"]),
            ]
        )
        + " |"
    )
    md.append(
        "| Random(seed=0) | "
        + " | ".join(
            [
                _fmt(stats_rand["mean"]),
                _fmt(stats_rand["median"]),
                _fmt(stats_rand["improved_frac"]),
                _fmt(stats_rand["p25"]),
                _fmt(stats_rand["p75"]),
                _fmt(stats_rand["min"]),
                _fmt(stats_rand["max"]),
                _fmt(stats_rand["n"]),
            ]
        )
        + " |"
    )

    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("=" * 80)
    print("[report]", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
