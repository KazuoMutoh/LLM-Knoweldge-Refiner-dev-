from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from simple_active_refine.util import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedArtifacts:
    artifacts_dir: Path
    entity_text: Dict[str, str]
    relation_text: Dict[str, str]


def _read_tsv_map(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        key = parts[0]
        val = parts[1]
        out[key] = val
    return out


def _iter_triples(path: Path) -> Iterable[Tuple[str, str, str]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        yield parts[0], parts[1], parts[2]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def prepare_simkgc_artifacts(
    *,
    dir_triples: Path,
    artifacts_dir: Path,
    entity2text_path: Optional[Path] = None,
    entity2textlong_path: Optional[Path] = None,
    relation2text_path: Optional[Path] = None,
) -> PreparedArtifacts:
    """Prepare SimKGC JSON artifacts from this repo's triple datasets.

    Expected input:
        - dir_triples/train.txt
        - dir_triples/valid.txt (optional)
        - dir_triples/test.txt (optional)
        - dir_triples/entity2textlong.txt or entity2text.txt (optional)
        - dir_triples/relation2text.txt (optional)

    Output files in artifacts_dir:
        - train.json / valid.json / test.json
        - entities.json
        - relation2text.txt (copied if present)

    Returns:
        PreparedArtifacts with loaded entity/relation text maps.
    """

    _ensure_dir(artifacts_dir)

    train_path = dir_triples / "train.txt"
    valid_path = dir_triples / "valid.txt"
    test_path = dir_triples / "test.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"train.txt not found: {train_path}")

    # Resolve text files.
    entity2textlong_path = entity2textlong_path or (dir_triples / "entity2textlong.txt")
    entity2text_path = entity2text_path or (dir_triples / "entity2text.txt")
    relation2text_path = relation2text_path or (dir_triples / "relation2text.txt")

    entity_text: Dict[str, str] = {}
    if entity2textlong_path.exists():
        entity_text = _read_tsv_map(entity2textlong_path)
    elif entity2text_path.exists():
        entity_text = _read_tsv_map(entity2text_path)

    relation_text: Dict[str, str] = {}
    if relation2text_path.exists():
        relation_text = _read_tsv_map(relation2text_path)
        # keep a copy for inspection
        (artifacts_dir / "relation2text.txt").write_text(
            relation2text_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    # Gather entities from triples.
    entity_ids: Set[str] = set()

    def build_split(split_path: Path) -> List[dict]:
        rows: List[dict] = []
        for h, r, t in _iter_triples(split_path):
            entity_ids.add(h)
            entity_ids.add(t)
            rows.append(
                {
                    "head_id": h,
                    "relation": r,
                    "tail_id": t,
                    "head": entity_text.get(h, h),
                    "tail": entity_text.get(t, t),
                }
            )
        return rows

    train_rows = build_split(train_path)
    _dump_json(artifacts_dir / "train.json", train_rows)

    if valid_path.exists() and valid_path.stat().st_size > 0:
        valid_rows = build_split(valid_path)
        _dump_json(artifacts_dir / "valid.json", valid_rows)

    if test_path.exists() and test_path.stat().st_size > 0:
        test_rows = build_split(test_path)
        _dump_json(artifacts_dir / "test.json", test_rows)

    entities_rows: List[dict] = []
    for eid in sorted(entity_ids):
        text = entity_text.get(eid, eid)
        # Split on first ':' as a simple name/desc convention.
        if ":" in text:
            name, desc = text.split(":", 1)
            entities_rows.append(
                {"entity_id": eid, "entity": name.strip(), "entity_desc": desc.strip()}
            )
        else:
            entities_rows.append({"entity_id": eid, "entity": text, "entity_desc": ""})

    _dump_json(artifacts_dir / "entities.json", entities_rows)

    logger.info("[simkgc] prepared artifacts at %s (entities=%d, train=%d)", artifacts_dir, len(entities_rows), len(train_rows))

    return PreparedArtifacts(
        artifacts_dir=artifacts_dir,
        entity_text={row["entity_id"]: (row["entity"] + (f": {row['entity_desc']}" if row["entity_desc"] else "")) for row in entities_rows},
        relation_text=relation_text,
    )
