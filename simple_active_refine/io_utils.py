from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set, Optional
import csv
import json

Triple = Tuple[str, str, str]


@dataclass
class KGPaths:
    """Paths for a triples directory.

    Attributes:
        root: Root directory containing KG files.
        train: Path to train.txt (tab‑separated h \t r \t t).
        valid: Path to valid.txt (optional).
        test: Path to test.txt (optional).
    """

    root: Path
    train: Path
    valid: Optional[Path] = None
    test: Optional[Path] = None

    @staticmethod
    def from_dir(dir_triples: str | Path) -> "KGPaths":
        root = Path(dir_triples)
        cand = {p.name.lower(): p for p in root.glob("*")}
        def pick(*names: str) -> Optional[Path]:
            for n in names:
                p = cand.get(n)
                if p and p.is_file():
                    return p
            return None
        train = pick("train.txt", "train.tsv")
        if train is None:
            raise FileNotFoundError("train.txt (or .tsv) not found in directory: " + str(root))
        valid = pick("valid.txt", "validation.txt", "dev.txt")
        test = pick("test.txt")
        return KGPaths(root=root, train=train, valid=valid, test=test)


def read_triples(path: str | Path) -> List[Triple]:
    """Read triples from a tab‑separated file.

    Args:
        path: Path to TSV file (h \t r \t t per line).

    Returns:
        List of triples as (head, relation, tail) strings.
    """
    triples: List[Triple] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                # tolerate space‑separated
                parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed triple line in {path}: {line}")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def write_triples(path: str | Path, triples: Iterable[Triple]) -> None:
    """Write triples to a tab‑separated file.

    Args:
        path: Output file path.
        triples: Iterable of (h, r, t).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def append_triples(path: str | Path, triples: Iterable[Triple]) -> None:
    """Append triples to an existing file (create if missing)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def load_kg(dir_triples: str | Path) -> Dict[str, List[Triple]]:
    """Load train/valid/test triples from directory.

    Args:
        dir_triples: Directory containing train.txt (and optionally valid/test).

    Returns:
        Dict with keys 'train','valid','test' mapping to triple lists (empty if missing).
    """
    paths = KGPaths.from_dir(dir_triples)
    data = {
        "train": read_triples(paths.train),
        "valid": read_triples(paths.valid) if paths.valid else [],
        "test": read_triples(paths.test) if paths.test else [],
    }
    return data


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)