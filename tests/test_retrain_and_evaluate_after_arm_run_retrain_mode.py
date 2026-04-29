from __future__ import annotations

import json
from pathlib import Path

from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.io_utils import write_triples

from retrain_and_evaluate_after_arm_run import run


def _write_tiny_dataset(dataset_dir: Path) -> None:
    # Small but non-trivial KG. Ensure test/valid exist.
    train = [
        ("a", "p", "b"),
        ("b", "p", "c"),
        ("c", "p", "d"),
        ("d", "p", "a"),
        ("a", "q", "c"),
        ("b", "q", "d"),
        ("c", "q", "a"),
        ("d", "q", "b"),
    ]
    valid = [("a", "p", "c"), ("b", "q", "a")]
    test = [("c", "p", "a"), ("d", "q", "c")]

    write_triples(dataset_dir / "train.txt", train)
    write_triples(dataset_dir / "valid.txt", valid)
    write_triples(dataset_dir / "test.txt", test)


def _write_embedding_config(path: Path) -> None:
    # Keep this minimal and fast. Ensure num_epochs is honored via top-level arg.
    cfg = {
        "model": "transe",
        "model_kwargs": {"embedding_dim": 8, "scoring_fct_norm": 1},
        "loss": "CrossEntropyLoss",
        "training_loop": "lcwa",
        "optimizer": "adam",
        "optimizer_kwargs": {"lr": 0.01, "weight_decay": 0.0},
        "training_kwargs": {"batch_size": 16, "label_smoothing": 0.0, "num_epochs": 2},
        "random_seed": 42,
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


def test_retrain_and_evaluate_after_arm_run_after_retrain_epoch2(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_tiny_dataset(dataset_dir)

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Minimal arm-run output with one iteration.
    (run_dir / "iter_1").mkdir()
    write_triples(
        run_dir / "iter_1" / "accepted_evidence_triples.tsv",
        [
            # Include one truly new triple (not in original train/valid/test)
            # while reusing existing entity/relation symbols so evaluation remains stable.
            ("a", "p", "d"),
            ("a", "q", "c"),
        ],
    )

    target_triples_path = tmp_path / "target_triples.txt"
    write_triples(target_triples_path, [("a", "p", "b"), ("b", "q", "d")])

    embedding_cfg_path = tmp_path / "embedding_config.json"
    _write_embedding_config(embedding_cfg_path)

    # Train BEFORE model for 2 epochs and save it.
    model_before_dir = tmp_path / "model_before"
    KnowledgeGraphEmbedding.train_model(
        model="transe",
        dir_triples=str(dataset_dir),
        dir_save=str(model_before_dir),
        model_kwargs={"embedding_dim": 8, "scoring_fct_norm": 1},
        loss="CrossEntropyLoss",
        training_loop="lcwa",
        optimizer="adam",
        optimizer_kwargs={"lr": 0.01, "weight_decay": 0.0},
        training_kwargs={"batch_size": 16, "label_smoothing": 0.0, "num_epochs": 2},
        random_seed=42,
    )

    # Run: AFTER should be retrained (epoch=2) on updated_triples.
    out_base = run(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        target_triples=target_triples_path,
        model_before_dir=model_before_dir,
        model_after_dir=None,
        exclude_predicate=None,
        after_mode="retrain",
        embedding_config=embedding_cfg_path,
        num_epochs=2,
        force_retrain=True,
    )

    model_after_dir = out_base / "model_after"
    assert (model_after_dir / "trained_model.pkl").exists()
    assert (model_after_dir / "training_triples").exists()

    # Ensure eval can find splits under model_after_dir.
    assert (model_after_dir / "test.txt").exists()
    assert (model_after_dir / "valid.txt").exists()

    # Outputs
    assert (out_base / "summary.json").exists()
    assert (out_base / "evaluation" / "iteration_metrics.json").exists()

    summary = json.loads((out_base / "summary.json").read_text(encoding="utf-8"))
    assert summary["after_mode"] == "retrain"
    assert summary["num_epochs"] == 2
    assert summary["updated_dataset"]["n_added_used"] == 1
