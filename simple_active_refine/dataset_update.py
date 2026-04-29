"""Dataset update utilities for arm-driven runs.

This module provides helpers to build an updated triples directory by
incorporating evidence triples accepted during an arm-driven refinement run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from simple_active_refine.io_utils import KGPaths, read_triples, write_triples
from simple_active_refine.util import get_logger

logger = get_logger(__name__)

Triple = Tuple[str, str, str]


@dataclass(frozen=True)
class EvidenceAggregationResult:
    """Result of aggregating accepted evidence triples from a run.

    Attributes:
        evidence_triples: Unique accepted evidence triples.
        n_iterations_seen: Number of iteration directories scanned.
        n_files_seen: Number of accepted evidence files found.
        n_triples_total: Total triples read before de-duplication.
    """

    evidence_triples: List[Triple]
    n_iterations_seen: int
    n_files_seen: int
    n_triples_total: int


@dataclass(frozen=True)
class UpdatedDatasetResult:
    """Result of creating an updated dataset directory.

    Attributes:
        out_dir: Output directory containing train/valid/test.
        n_train_before: Number of triples in the original train split.
        n_added_used: Number of evidence triples added (after exclusions and de-dup).
        n_train_after: Number of triples in the updated train split.
        excluded_predicates: Predicates excluded from evidence, if any.
    """

    out_dir: Path
    n_train_before: int
    n_added_used: int
    n_train_after: int
    excluded_predicates: Set[str]


def _iter_iteration_dirs(run_dir: Path) -> List[Path]:
    iter_dirs: List[Path] = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("iter_"):
            continue
        suffix = p.name[len("iter_") :]
        if not suffix.isdigit():
            continue
        iter_dirs.append(p)
    iter_dirs.sort(key=lambda d: int(d.name.split("_", 1)[1]))
    return iter_dirs


def aggregate_accepted_evidence_triples(run_dir: str | Path) -> EvidenceAggregationResult:
    """Aggregate unique accepted evidence triples from an arm-run directory.

    This reads `iter_*/accepted_evidence_triples.tsv` across the run directory.

    Args:
        run_dir: Arm-run output directory containing `iter_*` subdirectories.

    Returns:
        EvidenceAggregationResult: Aggregated evidence triples and counts.
    """

    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_path}")

    iter_dirs = _iter_iteration_dirs(run_path)
    n_iterations_seen = len(iter_dirs)

    all_triples: List[Triple] = []
    n_files_seen = 0
    for d in iter_dirs:
        f = d / "accepted_evidence_triples.tsv"
        if not f.exists():
            continue
        n_files_seen += 1
        triples = read_triples(f)
        all_triples.extend(triples)

    unique = sorted(set(all_triples))
    logger.info(
        "Aggregated accepted evidence: iter_dirs=%d files=%d total=%d unique=%d",
        n_iterations_seen,
        n_files_seen,
        len(all_triples),
        len(unique),
    )

    return EvidenceAggregationResult(
        evidence_triples=unique,
        n_iterations_seen=n_iterations_seen,
        n_files_seen=n_files_seen,
        n_triples_total=len(all_triples),
    )


def aggregate_accepted_added_triples(run_dir: str | Path) -> EvidenceAggregationResult:
    """Aggregate unique triples actually added to the KG from an arm-run directory.

    Preferred file is `iter_*/accepted_added_triples.tsv` (evidence + optional
    incident candidate triples). For backward compatibility with older runs,
    this falls back to `iter_*/accepted_evidence_triples.tsv` when the new file
    does not exist.

    Args:
        run_dir: Arm-run output directory containing `iter_*` subdirectories.

    Returns:
        EvidenceAggregationResult: Aggregated triples and counts.
    """

    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_path}")

    iter_dirs = _iter_iteration_dirs(run_path)
    n_iterations_seen = len(iter_dirs)

    all_triples: List[Triple] = []
    n_files_seen = 0

    for d in iter_dirs:
        f_new = d / "accepted_added_triples.tsv"
        f_old = d / "accepted_evidence_triples.tsv"
        f = f_new if f_new.exists() else f_old
        if not f.exists():
            continue
        n_files_seen += 1
        triples = read_triples(f)
        all_triples.extend(triples)

    unique = sorted(set(all_triples))
    logger.info(
        "Aggregated accepted added triples: iter_dirs=%d files=%d total=%d unique=%d",
        n_iterations_seen,
        n_files_seen,
        len(all_triples),
        len(unique),
    )

    return EvidenceAggregationResult(
        evidence_triples=unique,
        n_iterations_seen=n_iterations_seen,
        n_files_seen=n_files_seen,
        n_triples_total=len(all_triples),
    )


def _filter_by_excluded_predicates(triples: Sequence[Triple], excluded_predicates: Set[str]) -> List[Triple]:
    if not excluded_predicates:
        return list(triples)
    return [t for t in triples if t[1] not in excluded_predicates]


def create_updated_triples_dir(
    dataset_dir: str | Path,
    out_dir: str | Path,
    evidence_triples: Iterable[Triple],
    exclude_predicates: Optional[Sequence[str]] = None,
) -> UpdatedDatasetResult:
    """Create an updated triples directory by unioning train with evidence.

    Args:
        dataset_dir: Source dataset directory containing at least train.txt.
        out_dir: Output directory to write train/valid/test.
        evidence_triples: Evidence triples to add.
        exclude_predicates: Optional list of predicate strings to exclude from evidence.

    Returns:
        UpdatedDatasetResult: Summary of the updated dataset creation.
    """

    src_paths = KGPaths.from_dir(dataset_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    excluded = set(exclude_predicates or [])

    train_before = read_triples(src_paths.train)
    evidence_list = list(evidence_triples)
    evidence_filtered = _filter_by_excluded_predicates(evidence_list, excluded)

    set_train_before = set(train_before)
    set_evidence = set(evidence_filtered)
    set_train_after = set_train_before | set_evidence

    train_after = sorted(set_train_after)

    write_triples(out_path / "train.txt", train_after)

    # Copy valid/test content if present
    if src_paths.valid is not None:
        write_triples(out_path / "valid.txt", read_triples(src_paths.valid))
    if src_paths.test is not None:
        write_triples(out_path / "test.txt", read_triples(src_paths.test))

    logger.info(
        "Updated train triples: before=%d evidence_used=%d after=%d (excluded_predicates=%d)",
        len(set_train_before),
        len(set_train_after) - len(set_train_before),
        len(set_train_after),
        len(excluded),
    )

    return UpdatedDatasetResult(
        out_dir=out_path,
        n_train_before=len(set_train_before),
        n_added_used=len(set_train_after) - len(set_train_before),
        n_train_after=len(set_train_after),
        excluded_predicates=excluded,
    )
