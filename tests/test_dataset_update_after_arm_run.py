from pathlib import Path

from simple_active_refine.dataset_update import (
    aggregate_accepted_added_triples,
    aggregate_accepted_evidence_triples,
    create_updated_triples_dir,
)
from simple_active_refine.io_utils import write_triples, read_triples


def test_aggregate_accepted_evidence_triples(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "iter_1").mkdir()
    (run_dir / "iter_2").mkdir()

    write_triples(
        run_dir / "iter_1" / "accepted_evidence_triples.tsv",
        [("a", "p", "b"), ("x", "q", "y")],
    )
    write_triples(
        run_dir / "iter_2" / "accepted_evidence_triples.tsv",
        [("a", "p", "b"), ("c", "r", "d")],
    )

    agg = aggregate_accepted_evidence_triples(run_dir)

    assert agg.n_iterations_seen == 2
    assert agg.n_files_seen == 2
    assert agg.n_triples_total == 4
    assert set(agg.evidence_triples) == {("a", "p", "b"), ("x", "q", "y"), ("c", "r", "d")}


def test_aggregate_accepted_added_triples_prefers_new_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    (run_dir / "iter_1").mkdir()
    (run_dir / "iter_2").mkdir()

    # iter_1 has both: new file should be preferred
    write_triples(
        run_dir / "iter_1" / "accepted_evidence_triples.tsv",
        [("a", "p", "b")],
    )
    write_triples(
        run_dir / "iter_1" / "accepted_added_triples.tsv",
        [("a", "p", "b"), ("x", "q", "y")],
    )

    # iter_2 only has old file: should fallback
    write_triples(
        run_dir / "iter_2" / "accepted_evidence_triples.tsv",
        [("c", "r", "d")],
    )

    agg = aggregate_accepted_added_triples(run_dir)
    assert agg.n_iterations_seen == 2
    assert agg.n_files_seen == 2
    assert set(agg.evidence_triples) == {("a", "p", "b"), ("x", "q", "y"), ("c", "r", "d")}


def test_create_updated_triples_dir_exclude_predicate(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    write_triples(dataset_dir / "train.txt", [("a", "p", "b")])
    write_triples(dataset_dir / "valid.txt", [("v", "p", "w")])
    write_triples(dataset_dir / "test.txt", [("t", "p", "u")])

    out_dir = tmp_path / "updated"
    evidence = [("c", "r", "d"), ("e", "p", "f")]

    res = create_updated_triples_dir(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        evidence_triples=evidence,
        exclude_predicates=["p"],
    )

    train_out = read_triples(out_dir / "train.txt")
    assert set(train_out) == {("a", "p", "b"), ("c", "r", "d")}
    assert res.n_train_before == 1
    assert res.n_added_used == 1
    assert res.n_train_after == 2

    assert read_triples(out_dir / "valid.txt") == [("v", "p", "w")]
    assert read_triples(out_dir / "test.txt") == [("t", "p", "u")]
