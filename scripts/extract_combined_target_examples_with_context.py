"""Create combined rising/falling target examples (UCB+Random) with added-triple context.

User-facing intent:
- Do NOT separate examples by UCB vs Random.
- Do NOT use head/tail terminology.
- Still provide concrete added-triple context when available.

We interpret per-target 'delta' as minmax(train)-normalized score change (after - before)
computed in the existing analysis JSON.

For context, we show added triples whose *subject* equals either entity appearing in the
chosen target triple (i.e., matches one of the two entities in the target triple).

Usage:
  python /app/scripts/extract_combined_target_examples_with_context.py \
    --analysis_json /app/experiments/20260126_rerun1/analysis_ucb_vs_random_seed0_3relations_20260128.json \
    --out_md /app/experiments/20260126_rerun1/analysis_combined_examples_with_context_20260128.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


sys.path.insert(0, "/app")


Triple = Tuple[str, str, str]


def load_entity2text_map(dataset_dir: Path) -> Dict[str, str]:
    for name in ("entity2text.txt", "entity2textlong.txt"):
        p = dataset_dir / name
        if p.exists():
            mapping: Dict[str, str] = {}
            for line in p.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
            return mapping
    return {}


def read_triples(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split("\t") if "\t" in s else s.split()
        if len(parts) >= 3:
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def find_added_triples_path(summary_json_path: Path) -> Path:
    return summary_json_path.parent / "updated_triples" / "added_triples.tsv"


def triple_to_text(tr: Triple, entity2text: Dict[str, str]) -> str:
    a, r, b = tr
    return f"{entity2text.get(a, a)} — {r} — {entity2text.get(b, b)}"


def index_added_by_subject(added_triples: Iterable[Triple]) -> Dict[str, List[Triple]]:
    by_subj: Dict[str, List[Triple]] = defaultdict(list)
    for s, p, o in added_triples:
        by_subj[s].append((s, p, o))
    return by_subj


def collect_candidates(rel_payload: dict) -> List[dict]:
    out: List[dict] = []
    ex = rel_payload.get("examples", {})
    for method in ("ucb", "random"):
        for side in ("best", "worst"):
            for item in ex.get(method, {}).get(side, []) or []:
                out.append(item)
    # dedupe by triple
    seen = set()
    uniq: List[dict] = []
    for item in out:
        tr = tuple(item["triple"])
        if tr in seen:
            continue
        seen.add(tr)
        uniq.append(item)
    return uniq


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_json", type=Path, required=True)
    parser.add_argument("--out_md", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4, help="Number of rising/falling examples per relation")
    parser.add_argument("--k_added", type=int, default=3, help="Max context triples to show per example")
    args = parser.parse_args()

    data = json.loads(args.analysis_json.read_text())

    md: List[str] = []
    md.append("# Combined rising/falling target examples with added context (rerun1)")
    md.append("")
    md.append(f"- Source analysis JSON: `{args.analysis_json}`")
    md.append("")
    md.append(
        "Examples are selected from both UCB and Random(seed=0) runs, then ranked by Δ (minmax(train) normalized)."
    )
    md.append(
        "Added context is shown when an added triple's subject matches either entity in the target triple."
    )
    md.append("")

    for rel_name, rel_payload in data.items():
        paths = rel_payload.get("paths", {})
        dataset_dir = Path(paths["dataset_dir"])
        entity2text = load_entity2text_map(dataset_dir)

        # Load added triples from both methods, but we won't label which method they came from.
        ucb_added = read_triples(find_added_triples_path(Path(paths["ucb_summary"])))
        rand_added = read_triples(find_added_triples_path(Path(paths["random_summary"])))
        added_all = ucb_added + rand_added
        by_subj = index_added_by_subject(added_all)

        candidates = collect_candidates(rel_payload)
        candidates.sort(key=lambda x: float(x["delta"]))

        falling = candidates[: args.k]
        rising = list(reversed(candidates[-args.k :]))

        md.append(f"## {rel_name}")
        md.append("")

        def emit(title: str, items: List[dict]) -> None:
            md.append(f"### {title}")
            md.append("")
            for item in items:
                tr = tuple(item["triple"])
                delta = float(item["delta"])
                md.append(f"- Target: {triple_to_text(tr, entity2text)} (Δ={delta:+.3f})")

                # Context: subject matches either entity in the target triple
                a, _r, b = tr
                ctx = []
                for key in (a, b):
                    ctx.extend(by_subj.get(key, []))
                # de-dup and cap
                seen = set()
                uniq_ctx: List[Triple] = []
                for c in ctx:
                    if c in seen:
                        continue
                    seen.add(c)
                    uniq_ctx.append(c)
                uniq_ctx = uniq_ctx[: args.k_added]

                if uniq_ctx:
                    md.append("  - Added context (sample):")
                    for c in uniq_ctx:
                        md.append(f"    - {triple_to_text(c, entity2text)}")
                else:
                    md.append("  - Added context (sample): (none)")
            md.append("")

        emit("Rising examples", rising)
        emit("Falling examples", falling)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
