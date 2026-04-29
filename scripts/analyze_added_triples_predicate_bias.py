"""Analyze added-triple predicate types vs target score deltas.

This script deepens the UCB vs Random(seed=0) analysis by:
- computing per-target minmax(train)-normalized deltas (after - before)
- aggregating deltas by head entity (mean delta per head)
- comparing the *types* (predicates) of added triples overall vs those whose heads are
  associated with improved vs degraded target heads.

It is intended for analysis/records, not for the main pipeline.

Outputs:
- a Markdown report under the specified experiment directory

Assumptions:
- added triples are stored at:
  <run_dir>/arm_run/retrain_eval/updated_triples/added_triples.tsv
  (3-column TSV: head, relation, tail)

Usage:
  python /app/scripts/analyze_added_triples_predicate_bias.py \
    --analysis_json /app/experiments/20260126_rerun1/analysis_ucb_vs_random_seed0_3relations_20260128.json \
    --out_md /app/experiments/20260126_rerun1/analysis_added_triples_predicates_20260128.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# Allow running without setting PYTHONPATH
sys.path.insert(0, "/app")

from simple_active_refine.embedding import KnowledgeGraphEmbedding


Triple = Tuple[str, str, str]


def read_target_triples(path: Path) -> List[Triple]:
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


def read_added_triples_tsv(path: Path) -> List[Triple]:
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


def load_entity2text_map(dataset_dir: Path) -> Dict[str, str]:
    # Prefer entity2text.txt, fallback to entity2textlong.txt
    for name in ("entity2text.txt", "entity2textlong.txt"):
        p = dataset_dir / name
        if p.exists():
            mapping: Dict[str, str] = {}
            for line in p.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                mapping[parts[0]] = parts[1]
            return mapping
    return {}


def is_known(kge: KnowledgeGraphEmbedding, triple: Triple) -> bool:
    h, r, t = triple
    return (
        h in kge.triples.entity_to_id
        and t in kge.triples.entity_to_id
        and r in kge.triples.relation_to_id
    )


@dataclass(frozen=True)
class HeadDeltaSummary:
    head: str
    n_targets: int
    mean_delta: float
    improved_frac: float


@dataclass
class MethodAnalysis:
    kept_targets: List[Triple]
    target_deltas: List[float]
    head_summaries: Dict[str, HeadDeltaSummary]
    improved_heads: Set[str]
    degraded_heads: Set[str]


def compute_method_analysis(
    *,
    kge_before: KnowledgeGraphEmbedding,
    kge_after: KnowledgeGraphEmbedding,
    target_triples: Sequence[Triple],
) -> MethodAnalysis:
    kept: List[Triple] = []
    for tr in target_triples:
        if is_known(kge_before, tr) and is_known(kge_after, tr):
            kept.append(tr)

    if not kept:
        raise ValueError("No usable target triples after intersection filtering")

    before_scores = np.asarray(
        list(kge_before.score_triples(kept, normalize=True, norm_method="minmax")),
        dtype=float,
    )
    after_scores = np.asarray(
        list(kge_after.score_triples(kept, normalize=True, norm_method="minmax")),
        dtype=float,
    )
    deltas = (after_scores - before_scores).tolist()

    by_head: Dict[str, List[float]] = defaultdict(list)
    for (h, _r, _t), d in zip(kept, deltas):
        by_head[h].append(float(d))

    head_summaries: Dict[str, HeadDeltaSummary] = {}
    improved_heads: Set[str] = set()
    degraded_heads: Set[str] = set()

    for h, ds in by_head.items():
        arr = np.asarray(ds, dtype=float)
        mean_delta = float(np.mean(arr))
        improved_frac = float(np.mean(arr > 0.0))
        summary = HeadDeltaSummary(
            head=h,
            n_targets=int(arr.size),
            mean_delta=mean_delta,
            improved_frac=improved_frac,
        )
        head_summaries[h] = summary
        if mean_delta > 0.0:
            improved_heads.add(h)
        else:
            degraded_heads.add(h)

    return MethodAnalysis(
        kept_targets=kept,
        target_deltas=deltas,
        head_summaries=head_summaries,
        improved_heads=improved_heads,
        degraded_heads=degraded_heads,
    )


def counter_to_share(counter: Counter[str]) -> Dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def top_predicate_rows(
    *,
    counter_all: Counter[str],
    counter_pos: Counter[str],
    counter_neg: Counter[str],
    min_count: int = 5,
    top_k: int = 15,
) -> List[Tuple[str, int, float, float, float]]:
    """Return rows: (predicate, count_all, share_pos, share_neg, share_diff)."""

    share_pos = counter_to_share(counter_pos)
    share_neg = counter_to_share(counter_neg)

    candidates: List[Tuple[str, int, float, float, float]] = []
    for pred, cnt in counter_all.items():
        if cnt < min_count:
            continue
        sp = float(share_pos.get(pred, 0.0))
        sn = float(share_neg.get(pred, 0.0))
        candidates.append((pred, int(cnt), sp, sn, sp - sn))

    # Sort by absolute share difference (most skewed first)
    candidates.sort(key=lambda x: (abs(x[4]), x[1]), reverse=True)
    return candidates[:top_k]


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def triple_to_text(tr: Triple, entity2text: Dict[str, str]) -> str:
    h, r, t = tr
    ht = entity2text.get(h, h)
    tt = entity2text.get(t, t)
    return f"{ht} — {r} — {tt}"


def find_added_triples_path(summary_json_path: Path) -> Path:
    # The summary.json is in .../arm_run/retrain_eval/summary.json
    # added_triples.tsv is expected in .../arm_run/retrain_eval/updated_triples/added_triples.tsv
    return summary_json_path.parent / "updated_triples" / "added_triples.tsv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis_json",
        type=Path,
        required=True,
        help="Path to analysis_ucb_vs_random_seed0_3relations_*.json (contains model/target paths).",
    )
    parser.add_argument(
        "--out_md",
        type=Path,
        required=True,
        help="Output markdown path.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="How many skewed predicates to show.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Minimum predicate count in added triples to be considered.",
    )
    args = parser.parse_args()

    data = json.loads(args.analysis_json.read_text())

    md: List[str] = []
    md.append("# Added-triple predicate analysis (UCB vs Random seed=0)")
    md.append("")
    md.append(f"- Source analysis JSON: `{args.analysis_json}`")
    md.append("")
    md.append("This report asks: *what kinds of predicates are being added*, and are they skewed toward heads whose target triples improved vs degraded (based on minmax(train) Δ)?")
    md.append("")

    for rel_name, rel_payload in data.items():
        paths = rel_payload.get("paths", {})
        dataset_dir = Path(paths["dataset_dir"])
        target_path = Path(paths["target_triples"])
        before_dir = Path(paths["before_model"])
        ucb_after_dir = Path(paths["ucb_after_model"])
        rand_after_dir = Path(paths["random_after_model"])

        ucb_summary = Path(paths["ucb_summary"])
        rand_summary = Path(paths["random_summary"])

        ucb_added_path = find_added_triples_path(ucb_summary)
        rand_added_path = find_added_triples_path(rand_summary)

        md.append(f"## {rel_name}")
        md.append("")
        md.append(f"- target_triples: `{target_path}`")
        md.append(f"- before model: `{before_dir}`")
        md.append(f"- UCB added_triples: `{ucb_added_path}`")
        md.append(f"- Random added_triples: `{rand_added_path}`")
        md.append("")

        target_triples = read_target_triples(target_path)
        entity2text = load_entity2text_map(dataset_dir)

        # Load models once per relation
        kge_before = KnowledgeGraphEmbedding(model_dir=str(before_dir), fit_normalization=True)
        kge_ucb = KnowledgeGraphEmbedding(model_dir=str(ucb_after_dir), fit_normalization=True)
        kge_rand = KnowledgeGraphEmbedding(model_dir=str(rand_after_dir), fit_normalization=True)

        for label, kge_after, added_path in (
            ("UCB", kge_ucb, ucb_added_path),
            ("Random(seed=0)", kge_rand, rand_added_path),
        ):
            added_triples = read_added_triples_tsv(added_path)
            method = compute_method_analysis(
                kge_before=kge_before,
                kge_after=kge_after,
                target_triples=target_triples,
            )

            md.append(f"### {label}")
            md.append("")
            md.append(
                "- Head labels are computed by *mean* Δ across that head's target triples (mean Δ>0 => improved head)."
            )
            md.append(
                f"- Usable target triples: {len(method.kept_targets)} (unique heads: {len(method.head_summaries)})"
            )
            md.append(
                f"- Improved heads: {len(method.improved_heads)}, Degraded heads: {len(method.degraded_heads)}"
            )
            md.append(f"- Added triples: {len(added_triples)}")
            md.append("")

            c_all = Counter(r for _h, r, _t in added_triples)
            c_pos = Counter(r for h, r, _t in added_triples if h in method.improved_heads)
            c_neg = Counter(r for h, r, _t in added_triples if h in method.degraded_heads)

            md.append("#### Predicate distribution (overall)")
            md.append("")
            md.append("| rank | predicate | count |")
            md.append("|---:|---|---:|")
            for i, (pred, cnt) in enumerate(c_all.most_common(10), start=1):
                md.append(f"| {i} | {pred} | {cnt} |")
            md.append("")

            md.append("#### Predicates skewed to improved vs degraded heads")
            md.append("")
            md.append(
                f"(Showing predicates with count>= {args.min_count}, sorted by |share(improved) - share(degraded)|)"
            )
            md.append("")
            md.append("| predicate | count_all | share_improved | share_degraded | diff |")
            md.append("|---|---:|---:|---:|---:|")
            rows = top_predicate_rows(
                counter_all=c_all,
                counter_pos=c_pos,
                counter_neg=c_neg,
                min_count=args.min_count,
                top_k=args.top_k,
            )
            for pred, cnt, sp, sn, diff in rows:
                md.append(
                    "| "
                    + " | ".join(
                        [
                            pred,
                            str(cnt),
                            fmt_pct(sp),
                            fmt_pct(sn),
                            fmt_pct(diff),
                        ]
                    )
                    + " |"
                )
            if not rows:
                md.append("| (no predicates meet threshold) |  |  |  |  |")
            md.append("")

            # A couple of concrete examples to make it tangible
            # Pick one triple from improved-heads and one from degraded-heads, preferring common predicates.
            ex_pos: Optional[Triple] = None
            ex_neg: Optional[Triple] = None
            for tr in added_triples:
                h, r, t = tr
                if ex_pos is None and h in method.improved_heads:
                    ex_pos = tr
                if ex_neg is None and h in method.degraded_heads:
                    ex_neg = tr
                if ex_pos is not None and ex_neg is not None:
                    break

            md.append("#### Concrete examples (added triples)")
            md.append("")
            if ex_pos is not None:
                md.append(f"- Example (improved head): {triple_to_text(ex_pos, entity2text)}")
            else:
                md.append("- Example (improved head): (none found)")
            if ex_neg is not None:
                md.append(f"- Example (degraded head): {triple_to_text(ex_neg, entity2text)}")
            else:
                md.append("- Example (degraded head): (none found)")
            md.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
