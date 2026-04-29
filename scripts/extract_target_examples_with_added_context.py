"""Extract added-triple context for top improving/degrading target examples.

This script reads the existing rerun1 analysis JSON (per-target deltas + paths),
then for each relation/method extracts a few best/worst target triples and
attaches a small sample of added triples that share the same head.

Outputs a Markdown snippet (human-readable) suitable for pasting into a record.

Usage:
  python /app/scripts/extract_target_examples_with_added_context.py \
    --analysis_json /app/experiments/20260126_rerun1/analysis_ucb_vs_random_seed0_3relations_20260128.json \
    --out_md /app/experiments/20260126_rerun1/analysis_target_examples_with_added_context_20260128.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running without setting PYTHONPATH
sys.path.insert(0, "/app")


Triple = Tuple[str, str, str]


def read_added_triples(path: Path) -> List[Triple]:
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


def triple_to_text(tr: Triple, entity2text: Dict[str, str]) -> str:
    h, r, t = tr
    ht = entity2text.get(h, h)
    tt = entity2text.get(t, t)
    return f"{ht} — {r} — {tt}"


def find_added_triples_path(summary_json_path: Path) -> Path:
    return summary_json_path.parent / "updated_triples" / "added_triples.tsv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_json", type=Path, required=True)
    parser.add_argument("--out_md", type=Path, required=True)
    parser.add_argument("--k_examples", type=int, default=2, help="How many best/worst target examples to include.")
    parser.add_argument("--k_added", type=int, default=3, help="How many added triples to show per example head.")
    args = parser.parse_args()

    data = json.loads(args.analysis_json.read_text())

    md: List[str] = []
    md.append("# Target examples with added-triple context (rerun1)")
    md.append("")
    md.append(f"- Source analysis JSON: `{args.analysis_json}`")
    md.append("")

    for rel_name, rel_payload in data.items():
        paths = rel_payload.get("paths", {})
        dataset_dir = Path(paths["dataset_dir"])
        entity2text = load_entity2text_map(dataset_dir)

        md.append(f"## {rel_name}")
        md.append("")

        for method in ("ucb", "random"):
            ex = rel_payload.get("examples", {}).get(method)
            if not ex:
                continue

            summary_path = Path(paths[f"{method}_summary"])
            added_path = find_added_triples_path(summary_path)
            added = read_added_triples(added_path)

            # index by head for quick lookup
            by_head: Dict[str, List[Triple]] = {}
            for tr in added:
                by_head.setdefault(tr[0], []).append(tr)

            md.append(f"### {method}")
            md.append("")
            md.append(f"- added_triples: `{added_path}`")
            md.append("")

            def emit_block(title: str, items: List[dict]) -> None:
                md.append(f"#### {title}")
                md.append("")
                for item in items[: args.k_examples]:
                    tgt_triple = tuple(item["triple"])  # (h,r,t)
                    h = tgt_triple[0]
                    delta = float(item["delta"])
                    h_text = item.get("h_text") or entity2text.get(h, h)
                    t_text = item.get("t_text") or entity2text.get(tgt_triple[2], tgt_triple[2])
                    md.append(
                        f"- Target: {h_text} — {tgt_triple[1]} — {t_text} (Δ={delta:+.3f})"
                    )

                    ctx = by_head.get(h, [])[: args.k_added]
                    if ctx:
                        md.append("  - Added context (same head, sample):")
                        for tr in ctx:
                            md.append(f"    - {triple_to_text(tr, entity2text)}")
                    else:
                        md.append("  - Added context (same head, sample): (none)")
                md.append("")

            emit_block("best", ex.get("best", []))
            emit_block("worst", ex.get("worst", []))

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
