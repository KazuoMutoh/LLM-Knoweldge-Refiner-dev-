"""make_test_dataset.py

Create a synthetic test dataset by removing context triples around target entities.

This script is used to generate the datasets under `experiments/` such as
`experiments/test_data_for_nationality_v3`.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Set, Optional, Dict
import random
import json

from simple_active_refine.io_utils import load_kg, write_triples

Triple = Tuple[str, str, str]


def parse_entities(spec: str) -> List[str]:
    """Parse target entities from a comma‑separated string or a file path.

    If *spec* points to an existing file, it reads one entity id per line.
    Otherwise, it treats *spec* as a comma‑separated list.

    Args:
        spec: Path to file or comma‑separated string.

    Returns:
        List of entity identifiers (strings).
    """
    if spec in {"-", "auto", "AUTO", ""}:
        return []
    
    p = Path(spec)
    if p.exists():
        ents: List[str] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    ents.append(s)
        return ents
    
    return [s.strip() for s in spec.split(",") if s.strip()]


def _get_config_path_from_argv(argv: Optional[List[str]] = None) -> Optional[Path]:
    """Parse only --config from argv (if present).

    This allows config-driven defaults while keeping CLI override behavior.
    """
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", default=None)
    ns, _ = ap.parse_known_args(argv)
    if ns.config:
        return Path(ns.config)
    return None


def _load_config(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Backward-compatible aliases
    if "manifest" in config and "manifest_filename" not in config:
        config["manifest_filename"] = config["manifest"]
    if "selected_target_entities_filename" in config and "selected_target_entities_file" not in config:
        config["selected_target_entities_file"] = config["selected_target_entities_filename"]

    return config


def _count_entity_triples(base: List[Triple], entity: str) -> int:
    count = 0
    for h, _, t in base:
        if h == entity or t == entity:
            count += 1
    return count


def pick_neighbors(base: List[Triple], target_entities: Set[str], target_relation: str,
                   remove_preference: str) -> Set[str]:
    """Collect neighbor entities around target_entities via non‑target relations.

    Args:
        base: Triples from the base split.
        target_entities: Set of target entity ids.
        target_relation: Relation to exclude when collecting neighbors.
        remove_preference: 'head' (incoming), 'tail' (outgoing), or 'both'.

    Returns:
        Set of neighbor entity ids.
    """
    neighbors: Set[str] = set()
    use_in = remove_preference in {"head", "both"}
    use_out = remove_preference in {"tail", "both"}
    
    for h, r, t in base:
        if r == target_relation:
            continue
        if use_out and h in target_entities:
            neighbors.add(t)
        if use_in and t in target_entities:
            neighbors.add(h)
    
    return neighbors


def select_subset(items: List[str], ratio: float, rng: random.Random) -> Set[str]:
    """Sample a subset of *items* with proportion *ratio* (0..1)."""
    if ratio <= 0:
        return set()
    if ratio >= 1:
        return set(items)
    k = max(0, min(len(items), int(round(len(items) * ratio))))
    return set(rng.sample(items, k))


def compute_deletions(base: List[Triple], selected_entities: Set[str],
                       target_relation: str, include_target: bool) -> List[Triple]:
    """Compute triples to delete from *base* that involve *selected_entities*.

    Args:
        base: Base split triples.
        selected_entities: Entities whose incident triples will be deleted.
        target_relation: Target relation r_t.
        include_target: If False, keep triples with relation r_t.

    Returns:
        List of triples to delete.
    """
    deletions: List[Triple] = []
    for h, r, t in base:
        if (h in selected_entities) or (t in selected_entities):
            if include_target or r != target_relation:
                deletions.append((h, r, t))
    return deletions


def main():
    config_path = _get_config_path_from_argv()
    config = _load_config(config_path)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_triples", required=True)
    ap.add_argument("--dir_test_triples", required=True)
    ap.add_argument("--target_relation", required=True)
    ap.add_argument("--config", default=str(config_path) if config_path else "./config_dataset.json")

    ap.add_argument("--base_triples", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--target_entities", default="-")
    ap.add_argument("--auto_target_entities", type=int, default=0)
    ap.add_argument("--min_target_triples", type=int, default=0)
    ap.add_argument("--target_preference", default="head", choices=["head", "tail"])
    ap.add_argument("--remove_preference", default="both", choices=["head", "tail", "both"])
    ap.add_argument("--drop_ratio", type=float, default=0.7)
    ap.add_argument("--include_target", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--remove_target_incidents",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, remove incident triples of the target entities themselves (except target triples). "
            "If false, remove incident triples of sampled neighbor entities (default behavior)."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--removed_filename", default="removed.tsv")
    ap.add_argument("--manifest_filename", default="config_dataset.json")
    ap.add_argument("--selected_target_entities_filename", default="selected_target_entities.txt")

    ap.set_defaults(**config)
    args = ap.parse_args()

    # Normalize aliases for internal use/output parity
    if hasattr(args, "selected_target_entities_file") and not hasattr(args, "selected_target_entities_filename"):
        args.selected_target_entities_filename = getattr(args, "selected_target_entities_file")
    if hasattr(args, "manifest") and not hasattr(args, "manifest_filename"):
        args.manifest_filename = getattr(args, "manifest")

    if not (0.0 <= args.drop_ratio <= 1.0):
        raise SystemExit("drop_ratio must be in [0,1]")

    rng = random.Random(args.seed)

    # TODO: load_kg should be one of menber functions of KnowledgeGraphEmbedding in embedding.py
    data = load_kg(args.dir_triples)
    base = data[args.base_triples]

    # Parse or auto-select target entities
    # If the config includes a recorded selection (from a previous run), prefer it
    # when args.target_entities is '-' (auto).
    if args.target_entities in {"-", "auto", "AUTO", ""} and isinstance(config.get("target_entities_selected"), list):
        parsed_target_entities = [str(x) for x in config.get("target_entities_selected") if str(x).strip()]
    else:
        parsed_target_entities = parse_entities(args.target_entities)

    rng = random.Random(args.seed)

    if len(parsed_target_entities) == 0:
        if args.auto_target_entities <= 0:
            raise SystemExit(
                "target_entities not provided; either pass a list/file or use '-' with --auto_target_entities > 0"
            )

        triples_rt = [tr for tr in base if tr[1] == args.target_relation]
        rng.shuffle(triples_rt)
        side_idx = 0 if args.target_preference == "head" else 2

        picked: List[str] = []
        seen: Set[str] = set()
        for h, _, t in triples_rt:
            e = h if side_idx == 0 else t
            if e in seen:
                continue
            if args.min_target_triples > 0:
                if _count_entity_triples(base, e) < args.min_target_triples:
                    seen.add(e)
                    continue
            picked.append(e)
            seen.add(e)
            if len(picked) >= args.auto_target_entities:
                break
        parsed_target_entities = picked

        if len(parsed_target_entities) == 0:
            raise SystemExit(
                "No candidate entities found for automatic selection. Check target_relation and base_triples."
            )

    target_entities = set(parsed_target_entities)

    # -------------------------------------------
    # 1) Extract target triples S (for information/debug; not strictly required downstream)
    # -------------------------------------------
    if args.target_preference == "head":
        target_triples = [(h, r, t) for (h, r, t) in base if r == args.target_relation and h in target_entities]
    else:
        target_triples = [(h, r, t) for (h, r, t) in base if r == args.target_relation and t in target_entities]

    # -------------------------------------------
    # 2) Choose deletion seed entities
    # -------------------------------------------
    # Default behavior: remove context around sampled neighbor entities.
    # New behavior: remove incident triples of the target entities themselves.
    removal_mode = "target_incidents" if args.remove_target_incidents else "neighbors"

    if args.remove_target_incidents:
        neighbors_all: Set[str] = set()
        neighbors_sel: Set[str] = set()
        deletion_seed_entities = set(target_entities)
    else:
        # Collect neighbors via non‑target relations, then sample with drop_ratio.
        # If config provides pre-selected neighbors, use them for exact reproduction.
        if args.target_entities in {"-", "auto", "AUTO", ""} and isinstance(config.get("neighbors_selected"), list):
            neighbors_all = set(str(x) for x in (config.get("neighbors_all") or []) if str(x).strip())
            neighbors_sel = set(str(x) for x in config.get("neighbors_selected") if str(x).strip())
        else:
            neighbors_all = pick_neighbors(base, target_entities, args.target_relation, args.remove_preference)
            neighbors_sel = select_subset(sorted(neighbors_all), args.drop_ratio, rng)
        deletion_seed_entities = neighbors_sel

    # -------------------------------------------
    # 3) Deletion triples are those in base incident to deletion_seed_entities
    # -------------------------------------------
    removed_triples = compute_deletions(base, deletion_seed_entities, args.target_relation, args.include_target)
    target_triple_set = set(target_triples)
    if target_triple_set:
        removed_triples = [tr for tr in removed_triples if tr not in target_triple_set]


    # -------------------------------------------
    # 4) Create a new dataset
    # -------------------------------------------
    new_data: Dict[str, Dict[str, List[Triple]]] = {}

    removed_set = set(removed_triples)
    new_data[args.base_triples] = {
        "removed": list(removed_triples),
        "included": [tr for tr in base if tr not in removed_set],
    }

    entities_removed: Set[str] = set()
    for h, _, t in removed_triples:
        entities_removed.add(h)
        entities_removed.add(t)

    for split in ("train", "valid", "test"):
        if split == args.base_triples:
            continue
        included_split: List[Triple] = []
        removed_split: List[Triple] = []
        for h, r, t in data.get(split, []):
            if h in entities_removed or t in entities_removed:
                removed_split.append((h, r, t))
            else:
                included_split.append((h, r, t))
        new_data[split] = {"removed": removed_split, "included": included_split}

    mapping: Dict[str, List[Triple]] = {}
    for split in ("train", "valid", "test"):
        for h, r, t in new_data.get(split, {}).get("removed", []):
            if h in target_entities:
                mapping.setdefault(h, []).append((h, r, t))
            if t in target_entities:
                mapping.setdefault(t, []).append((h, r, t))

    # -------------------------------------------
    # 5) Save to dir_test_triples as a new dataset
    # -------------------------------------------

    # write triples in output directory
    # files for included triples should be train.txt, test.txt, valid.txt.
    # files for removed triples should be train_removed.txt, test_removed.txt, valid_removed.txt

    out = Path(args.dir_test_triples)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        included = new_data.get(split, {}).get('included', [])
        removed = new_data.get(split, {}).get('removed', [])
        # Write included triples
        write_triples(out / f"{split}.txt", included)
        # Write removed triples
        write_triples(out / f"{split}_removed.txt", removed)

    # Save selected target entities
    with open(out / args.selected_target_entities_filename, "w", encoding="utf-8") as f:
        for ent in sorted(target_entities):
            f.write(ent + "\n")

    # Save target triples.
    with open(out / "target_triples.txt", "w", encoding="utf-8") as f:
        for h, r, t in target_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    # save mapping in output directory as json
    with open(out / "entity_removed_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # For manifest, compute deletions and new_base for reporting
    deletions = new_data[args.base_triples]["removed"]
    new_base = new_data[args.base_triples]["included"]

    config_dataset = {
        "dir_triples": str(Path(args.dir_triples).resolve()),
        "dir_test_triples": str(Path(args.dir_test_triples).resolve()),
        "base_triples": args.base_triples,
        "target_entities": args.target_entities,
        "target_relation": args.target_relation,
        "target_preference": args.target_preference,
        "remove_preference": args.remove_preference,
        "drop_ratio": args.drop_ratio,
        "include_target": args.include_target,
        "remove_target_incidents": args.remove_target_incidents,
        "removal_mode": removal_mode,
        "seed": args.seed,
        "removed_filename": args.removed_filename,
        "manifest_filename": args.manifest_filename,
        "auto_target_entities": args.auto_target_entities,
        "selected_target_entities_file": args.selected_target_entities_filename,
        "n_base_in": len(base),
        "n_removed": len(deletions),
        "n_base_out": len(new_base),
        "n_target_triples": len(target_triples),
        "n_neighbors_all": len(neighbors_all),
        "n_neighbors_selected": len(neighbors_sel),
        "n_target_entities_selected": len(target_entities),
        "target_entities_selected": sorted(target_entities),
        "target_triples": target_triples,
        "neighbors_all": sorted(neighbors_all),
        "neighbors_selected": sorted(neighbors_sel),
    }
    with open(out / "config_dataset.json", 'w') as fout:
        json.dump(config_dataset, fout, indent=2, ensure_ascii=False)

    print(
        f"Created test dataset at {out}. "
        f"base={args.base_triples} in={len(base)} removed={len(deletions)} out={len(new_base)}"
    )


if __name__ == "__main__":
    main()