# =============================
# file: make_test_triples.py
# =============================
from __future__ import annotations
"""
Create a synthetic test dataset by removing context triples around target entities.

Mathematical sketch
-------------------
Given a KG G = (E, R, G) with splits (G_train, G_valid, G_test), a target relation r_t
and a set of target entities T ⊆ E. Let B ∈ {train, valid, test} be the base split.
Let P ∈ {head, tail} denote the role of target entities within r_t (target_preference),
let Q ∈ {head, tail, both} denote the removal preference over non‑target relations.

1) Extract target triples S = { (h, r_t, t) ∈ G_B | (P=head ∧ h∈T) ∨ (P=tail ∧ t∈T) }.
2) For each e∈T, collect neighbor entities via non‑target relations:
   N_in(e)  = { u | (u, r, e) ∈ G_B, r≠r_t }, N_out(e) = { v | (e, r, v) ∈ G_B, r≠r_t }.
   If Q=head → use N_in(e); Q=tail → use N_out(e); Q=both → use N_in(e) ∪ N_out(e).
   Sample a subset R_e ⊆ N_* (drop_ratio proportion).
3) Let R = ⋃_e R_e be selected neighbor entities. Define deletion set D as:
   D = { (h,r,t) ∈ G_B | (h∈R or t∈R) and (include_target or r≠r_t) }.
4) Output a new dataset where base split becomes G_B' = G_B \ D, while other splits
   are copied as is. Also persist D as removed.tsv and a manifest JSON.

This procedure removes auxiliary context around the target_entities while keeping the
option to preserve target‑relation triples intact (default).
"""
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Set
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


def count_entity_triples(base: List[Triple], entity: str) -> int:
    """Count the number of triples involving the given entity.
    
    Args:
        base: Base split triples.
        entity: Entity to count triples for.
    
    Returns:
        Number of triples containing the entity.
    """
    count = 0
    for h, r, t in base:
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
            print(h,r,t)
        if use_in and t in target_entities:
            neighbors.add(h)
            print(h,r,t)
    
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
        include_target: If True, also delete triples with relation r_t for selected_entities.
                       If False, keep triples with relation r_t for selected_entities.

    Returns:
        List of triples to delete.
    """
    deletions: List[Triple] = []
    
    # Collect triples incident to selected_entities
    for h, r, t in base:
        if (h in selected_entities) or (t in selected_entities):
            if include_target or r != target_relation:
                deletions.append((h, r, t))
    
    print(f"Total deletions: {len(deletions)}")
    return deletions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_triples", required=True)
    ap.add_argument("--dir_test_triples", required=True)
    ap.add_argument("--target_relation", required=True)
    ap.add_argument("--config", default='./config_dataset.json')
    # Load additional configuration from dataset_config.json
    args = ap.parse_args()
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    ap.set_defaults(**config)
    args = ap.parse_args()

    if not (0.0 <= args.drop_ratio <= 1.0):
        raise SystemExit("drop_ratio must be in [0,1]")

    rng = random.Random(args.seed)

    # TODO: load_kg should be one of menber functions of KnowledgeGraphEmbedding in embedding.py
    data = load_kg(args.dir_triples)
    base = data[args.base_triples]

    # Parse or auto-select target entities
    parsed_target_entities = parse_entities(args.target_entities)
    rng = random.Random(args.seed)

    if len(parsed_target_entities) == 0:
        if args.auto_target_entities <= 0:
            raise SystemExit("target_entities not provided; either pass a list/file or use '-' with --auto-target_entities > 0")
        # Build candidate entities from target_relation triples according to preference
        triples_rt = [tr for tr in base if tr[1] == args.target_relation]
        rng.shuffle(triples_rt)
        side_idx = 0 if args.target_preference == "head" else 2
        picked, seen = [], set()
        
        print(f"Filtering candidates with minimum {args.min_target_triples} triples...")
        for h, r, t in triples_rt:
            e = h if side_idx == 0 else t
            if e not in seen:
                # Check if entity has enough context (minimum number of triples)
                triple_count = count_entity_triples(base, e)
                if triple_count >= args.min_target_triples:
                    picked.append(e)
                    seen.add(e)
                else:
                    print(f"  Skipping {e}: only {triple_count} triples (< {args.min_target_triples})")
            if len(picked) >= args.auto_target_entities:
                break
        parsed_target_entities = picked
        if len(parsed_target_entities) == 0:
            raise SystemExit("No candidate entities found for automatic selection. Check target_relation and base_triples.")

    target_entities = set(parsed_target_entities)
    desired_count = len(target_entities)  # Remember the original desired count

    # -------------------------------------------
    # Filter out target entities that are neighbors of other target entities
    # and refill to maintain desired count
    # -------------------------------------------
    print(f"\nChecking for mutual neighbor relationships among target entities...")
    
    # Build a list of all candidates (for refilling)
    if len(parsed_target_entities) > 0 and args.auto_target_entities > 0:
        triples_rt = [tr for tr in base if tr[1] == args.target_relation]
        side_idx = 0 if args.target_preference == "head" else 2
        all_candidates = []
        seen_candidates = set()
        for h, r, t in triples_rt:
            e = h if side_idx == 0 else t
            if e not in seen_candidates:
                triple_count = count_entity_triples(base, e)
                if triple_count >= args.min_target_triples:
                    all_candidates.append(e)
                    seen_candidates.add(e)
    else:
        all_candidates = []
    
    max_iterations = 10  # Prevent infinite loop
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Build neighbor graph: which target entities are connected via non-target relations
        mutual_neighbors = set()
        for h, r, t in base:
            if r == args.target_relation:
                continue
            # If both h and t are target entities, they are mutual neighbors
            if h in target_entities and t in target_entities:
                # Prefer to remove the one with fewer triples
                count_h = count_entity_triples(base, h)
                count_t = count_entity_triples(base, t)
                if count_h < count_t:
                    mutual_neighbors.add(h)
                elif count_t < count_h:
                    mutual_neighbors.add(t)
                else:
                    # If equal, remove one arbitrarily (deterministic by entity id)
                    if h < t:
                        mutual_neighbors.add(h)
                    else:
                        mutual_neighbors.add(t)
        
        if len(mutual_neighbors) == 0:
            if iteration == 1:
                print(f"  No mutual neighbor relationships found. All {len(target_entities)} targets are independent.")
            else:
                print(f"  Iteration {iteration}: No more mutual neighbors. Final count: {len(target_entities)}")
            break
        
        print(f"  Iteration {iteration}: Found {len(mutual_neighbors)} target entities that are neighbors of other targets")
        print(f"  Removing them to avoid complete context loss...")
        for entity in list(mutual_neighbors)[:5]:
            print(f"    - {entity}")
        if len(mutual_neighbors) > 5:
            print(f"    ... and {len(mutual_neighbors) - 5} more")
        
        target_entities = target_entities - mutual_neighbors
        print(f"  Remaining target entities: {len(target_entities)}")
        
        # Refill to maintain desired count
        num_to_add = desired_count - len(target_entities)
        if num_to_add > 0 and len(all_candidates) > 0:
            print(f"  Refilling with {num_to_add} new candidates...")
            added = 0
            for candidate in all_candidates:
                if candidate not in target_entities and candidate not in mutual_neighbors:
                    target_entities.add(candidate)
                    added += 1
                    if added >= num_to_add:
                        break
            print(f"  Added {added} new targets. Current count: {len(target_entities)}")
            
            if added < num_to_add:
                print(f"  Warning: Could only add {added}/{num_to_add} targets. No more valid candidates available.")
                break
        elif num_to_add > 0:
            print(f"  Warning: Need to add {num_to_add} more targets but no candidates available.")
            break
        else:
            # Already at desired count, continue checking for mutual neighbors
            pass
    
    if iteration >= max_iterations:
        print(f"  Warning: Reached maximum iterations ({max_iterations}). Stopping mutual neighbor removal.")
    
    print(f"\nFinal target entity count: {len(target_entities)} (desired: {desired_count})")

    # -------------------------------------------
    # 1) Extract target triples S (for information/debug; not strictly required downstream)
    # -------------------------------------------
    if args.target_preference == "head":
        target_triples = [(h, r, t) for (h, r, t) in base if r == args.target_relation and h in target_entities]
    else:
        target_triples = [(h, r, t) for (h, r, t) in base if r == args.target_relation and t in target_entities]

    # -------------------------------------------
    # 2) Collect neighbors via non‑target relations, then sample with drop_ratio
    # -------------------------------------------
    neighbors_all = pick_neighbors(base, target_entities, args.target_relation, args.remove_preference)
    neighbors_sel = select_subset(sorted(neighbors_all), args.drop_ratio, rng)

    # -------------------------------------------
    # 3) Deletion triples are those in base incident to selected neighbors
    # -------------------------------------------
    removed_triples = compute_deletions(
        base, 
        neighbors_sel, 
        args.target_relation, 
        args.include_target
    )


    # -------------------------------------------
    # 4) Create a new dataset
    # -------------------------------------------
    new_data = dict()
    
    # remove from base
    base_set = set(base)
    removed_triples = set(removed_triples)
    included_triples = [tr for tr in base_set if tr not in removed_triples]

    new_data[args.base_triples] = {'removed':removed_triples, 'included': included_triples}

    entities_removed = set()
    for h, r, t in removed_triples:
        entities_removed.add(h)
        entities_removed.add(t)

    # remove from other split
    for split in ("train", "valid", "test"):
        if split == args.base_triples:
            continue
        if data[split]:
            included_triples = []
            removed_triples = []
            for h, r, t in data[split]:
                if h in entities_removed or t in entities_removed:
                    removed_triples.append((h,r,t))
                else:
                    included_triples.append((h,r,t))
            new_data[split] = {'removed':removed_triples, 'included': included_triples}

    mapping = dict()
    for split in {'train', 'valid', 'test'}:
        if split not in new_data:
            continue
        removed_triples = new_data[split]['removed']
        for triple in removed_triples:
            
            if triple[0] in target_entities:
                target_entity = triple[0]
            if triple[2] in target_entities:
                target_entity = triple[2]
            else:
                continue

            if target_entity not in mapping.keys():
                mapping[target_entity] = []
            mapping[target_entity].append(triple) 

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
    deletions = new_data[args.base_triples]['removed']
    new_base = new_data[args.base_triples]['included']

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
        "seed": args.seed,
        "removed_filename": args.removed_filename,
        "manifest_filename": args.manifest,
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

    print(f"Created test dataset at {out}. Removed {len(removed_triples)} triples from {args.base_triples}.")


if __name__ == "__main__":
    main()
