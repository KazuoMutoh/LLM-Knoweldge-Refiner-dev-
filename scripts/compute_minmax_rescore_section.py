"""Compute minmax(train) normalized target rescoring deltas and write a markdown section.

This is used to compare UCB vs Random(seed0..3) for KG-FIT(PairRE).

It scores the *same* target triples with:
- fixed PairRE before model
- each PairRE after model

Then computes Δ = after_norm - before_norm where norm is per-model minmax(train).
To make the comparison consistent, it uses the intersection of target triples
known to *all* models in the comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Allow running without setting PYTHONPATH
sys.path.insert(0, "/app")

from simple_active_refine.embedding import KnowledgeGraphEmbedding


Triple = Tuple[str, str, str]


def read_target(path: Path) -> List[Triple]:
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


def is_known(kge: KnowledgeGraphEmbedding, triple: Triple) -> bool:
    h, r, t = triple
    return (
        h in kge.triples.entity_to_id
        and t in kge.triples.entity_to_id
        and r in kge.triples.relation_to_id
    )


def delta_stats(before: List[float], after: List[float]) -> dict:
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


def fmt(x) -> str:
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def load_summary(p: Path) -> dict:
    return json.loads(p.read_text())


def main() -> int:
    ucb_summary = Path(
        "/app/experiments/20260122/exp_ucb_arm10_priors_off_kgfit_pairre_ep100_20260122b/arm_run/retrain_eval/summary.json"
    )
    rand_summaries = {
        0: Path(
            "/app/experiments/20260123/exp_random_add10232_priors_off_kgfit_pairre_ep100_seed0_20260123a/arm_run/retrain_eval/summary.json"
        ),
        1: Path(
            "/app/experiments/20260123/exp_random_add10232_priors_off_kgfit_pairre_ep100_seed1_20260123a/arm_run/retrain_eval/summary.json"
        ),
        2: Path(
            "/app/experiments/20260123/exp_random_add10232_priors_off_kgfit_pairre_ep100_seed2_20260123a/arm_run/retrain_eval/summary.json"
        ),
        3: Path(
            "/app/experiments/20260123/exp_random_add10232_priors_off_kgfit_pairre_ep100_seed3_20260123a/arm_run/retrain_eval/summary.json"
        ),
    }

    u = load_summary(ucb_summary)
    before_dir = Path(u["model_before_dir"])
    target_path = Path(u["target_triples"])

    triples = read_target(target_path)
    if not triples:
        raise SystemExit(f"No target triples found: {target_path}")

    kge_before = KnowledgeGraphEmbedding(model_dir=str(before_dir), fit_normalization=True)

    runs = [("UCB", Path(u["model_after_dir"]))]
    for seed, sp in sorted(rand_summaries.items()):
        s = load_summary(sp)
        runs.append((f"Random(seed={seed})", Path(s["model_after_dir"])) )

    kges_after = [
        (name, KnowledgeGraphEmbedding(model_dir=str(after_dir), fit_normalization=True))
        for name, after_dir in runs
    ]

    kept: List[Triple] = []
    skipped: List[Triple] = []
    for tr in triples:
        if is_known(kge_before, tr) and all(is_known(kge, tr) for _n, kge in kges_after):
            kept.append(tr)
        else:
            skipped.append(tr)

    before_scores = list(kge_before.score_triples(kept, normalize=True, norm_method="minmax"))

    rows = []
    for name, kge_after in kges_after:
        after_scores = list(kge_after.score_triples(kept, normalize=True, norm_method="minmax"))
        rows.append((name, delta_stats(before_scores, after_scores)))

    rand_means = [d["mean"] for name, d in rows if name.startswith("Random(")]

    md: List[str] = []
    md.append("## Minmax(train) normalized target rescoring")
    md.append("")
    md.append(
        "Target triples are re-scored with the fixed PairRE-before model and each PairRE-after model, "
        "using per-model minmax(train) normalization; Δ is computed on the intersection of target triples "
        "known to *all* models in this comparison."
    )
    md.append("")
    md.append(f"- target_triples: `{target_path}`")
    md.append(f"- before model: `{before_dir}`")
    md.append(
        f"- usable targets (intersection): {len(kept)} (skipped {len(skipped)} / total {len(triples)})"
    )
    md.append("")
    md.append("| condition | Δmean | Δmedian | improved_frac | p25 | p75 | min | max | n |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in rows:
        md.append(
            "| "
            + " | ".join(
                [
                    name,
                    fmt(d["mean"]),
                    fmt(d["median"]),
                    fmt(d["improved_frac"]),
                    fmt(d["p25"]),
                    fmt(d["p75"]),
                    fmt(d["min"]),
                    fmt(d["max"]),
                    str(d["n"]),
                ]
            )
            + " |"
        )

    if rand_means:
        md.append("")
        md.append(
            f"- Random(seed0..3) Δmean(minmax) aggregate: mean={np.mean(rand_means):.6g}, std={np.std(rand_means):.6g}"
        )

    out = Path("/app/experiments/20260123/minmax_target_rescore_section_seed0_3.md")
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
