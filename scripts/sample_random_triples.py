"""Sample triples from a TSV file (3 columns) deterministically.

This script is intended to create `accepted_added_triples.tsv` compatible with
`retrain_and_evaluate_after_arm_run.py`.

Sampling is performed *without replacement* (no duplicate lines in the output).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


Triple = Tuple[str, str, str]


def _read_triples_tsv(path: Path) -> List[Triple]:
    """Read 3-column TSV triples.

    Args:
        path: Path to TSV file with 3 columns (head, relation, tail).

    Returns:
        List of triples.

    Raises:
        ValueError: If any line does not have exactly 3 columns.
    """

    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid TSV at {path} line {line_no}: expected 3 cols, got {len(parts)}"
                )
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def _write_triples_tsv(path: Path, triples: Sequence[Triple]) -> None:
    """Write 3-column TSV triples."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def sample_triples_without_replacement(
    triples: Sequence[Triple],
    n: int,
    seed: int,
) -> List[Triple]:
    """Sample N triples without replacement.

    Args:
        triples: Candidate triples.
        n: Number of triples to sample.
        seed: Random seed (deterministic).

    Returns:
        Sampled list of triples.

    Raises:
        ValueError: If n is larger than the number of available unique triples.
    """

    unique_triples = list(dict.fromkeys(triples))
    if n > len(unique_triples):
        raise ValueError(f"n={n} is larger than available unique triples={len(unique_triples)}")

    rng = random.Random(seed)
    return rng.sample(unique_triples, n)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample N triples from a 3-column TSV (without replacement)."
    )
    parser.add_argument(
        "--input_tsv",
        required=True,
        type=Path,
        help="Input TSV path (3 columns: head, relation, tail).",
    )
    parser.add_argument(
        "--output_tsv",
        required=True,
        type=Path,
        help="Output TSV path (3 columns).",
    )
    parser.add_argument("--n", required=True, type=int, help="Number of triples to sample.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    triples = _read_triples_tsv(args.input_tsv)
    sampled = sample_triples_without_replacement(triples, n=args.n, seed=args.seed)
    _write_triples_tsv(args.output_tsv, sampled)

    print(
        f"Wrote {len(sampled)} triples to {args.output_tsv} "
        f"(input={args.input_tsv}, seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
