from __future__ import annotations

from pathlib import Path

import pytest

import run_full_arm_pipeline as runner


class _StubPipe:
    def __init__(self, arm_run_dir: Path):
        self._arm_run_dir = arm_run_dir

    def run(self) -> None:
        # Minimal arm-run output that downstream expects.
        it = self._arm_run_dir / "iter_1"
        it.mkdir(parents=True, exist_ok=True)
        (it / "accepted_evidence_triples.tsv").write_text("a\tp\tb\n", encoding="utf-8")


def test_run_pipeline_calls_steps_and_writes_expected_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.txt").write_text("a\tp\tb\n", encoding="utf-8")
    (dataset_dir / "valid.txt").write_text("a\tp\tb\n", encoding="utf-8")
    (dataset_dir / "test.txt").write_text("a\tp\tb\n", encoding="utf-8")

    target_triples = tmp_path / "target_triples.txt"
    target_triples.write_text("a\tp\tb\n", encoding="utf-8")

    candidate_triples = tmp_path / "candidate_triples.txt"
    candidate_triples.write_text("a\tp\tb\n", encoding="utf-8")

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    calls: list[str] = []

    def _stub_build_rule_pool(**kwargs):
        calls.append("rule_pool")
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "initial_rule_pool.pkl").write_bytes(b"stub")
        (out_dir / "initial_rule_pool.csv").write_text("stub", encoding="utf-8")
        (out_dir / "initial_rule_pool.txt").write_text("stub", encoding="utf-8")
        return (
            str(out_dir / "initial_rule_pool.csv"),
            str(out_dir / "initial_rule_pool.pkl"),
            str(out_dir / "initial_rule_pool.txt"),
        )

    def _stub_build_arms(**kwargs):
        calls.append("arms")
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "initial_arms.json").write_text("{}", encoding="utf-8")
        (out_dir / "initial_arms.pkl").write_bytes(b"stub")
        (out_dir / "initial_arms.txt").write_text("stub", encoding="utf-8")
        return (
            str(out_dir / "initial_arms.json"),
            str(out_dir / "initial_arms.pkl"),
            str(out_dir / "initial_arms.txt"),
        )

    def _stub_from_paths(*, config, initial_arms_path, rule_pool_pkl, dir_triples, target_triples_path, candidate_triples_path):
        calls.append("arm_run")
        # Ensure paths are chained correctly.
        assert Path(initial_arms_path).name == "initial_arms.json"
        assert Path(rule_pool_pkl).name == "initial_rule_pool.pkl"
        assert Path(dir_triples) == dataset_dir
        assert Path(target_triples_path) == target_triples
        assert Path(candidate_triples_path) == candidate_triples
        return _StubPipe(Path(config.base_output_path))

    def _stub_retrain_eval(**kwargs):
        calls.append("eval")
        arm_run_dir = Path(kwargs["run_dir"])
        out_base = arm_run_dir / "retrain_eval"
        out_base.mkdir(parents=True, exist_ok=True)
        (out_base / "summary.json").write_text("{}", encoding="utf-8")
        return out_base

    monkeypatch.setattr(runner, "build_initial_rule_pool", _stub_build_rule_pool)
    monkeypatch.setattr(runner, "build_initial_arm_pool", _stub_build_arms)
    monkeypatch.setattr(runner, "ArmDrivenKGRefinementPipeline", type("P", (), {"from_paths": staticmethod(_stub_from_paths)}))
    monkeypatch.setattr(runner, "retrain_and_evaluate", _stub_retrain_eval)

    paths = runner.run_pipeline(
        run_dir=run_dir,
        model_dir=model_dir,
        target_relation="/r",
        dataset_dir=dataset_dir,
        target_triples=target_triples,
        candidate_triples=candidate_triples,
        force=False,
    )

    assert calls == ["rule_pool", "arms", "arm_run", "eval"]
    assert paths.rule_pool_pkl.exists()
    assert paths.arms_json.exists()
    assert paths.retrain_eval_summary.exists()


def test_run_pipeline_skips_existing_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "train.txt").write_text("a\tp\tb\n", encoding="utf-8")

    target_triples = tmp_path / "target_triples.txt"
    target_triples.write_text("a\tp\tb\n", encoding="utf-8")

    candidate_triples = tmp_path / "candidate_triples.txt"
    candidate_triples.write_text("a\tp\tb\n", encoding="utf-8")

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    paths = runner._resolve_paths(run_dir)
    paths.rule_pool_dir.mkdir(parents=True, exist_ok=True)
    paths.arms_dir.mkdir(parents=True, exist_ok=True)
    paths.arm_run_dir.mkdir(parents=True, exist_ok=True)

    paths.rule_pool_pkl.write_bytes(b"stub")
    paths.arms_json.write_text("{}", encoding="utf-8")

    # Pretend arm-run already happened.
    (paths.arm_run_dir / "iter_1").mkdir(parents=True, exist_ok=True)

    # Pretend eval already happened.
    paths.retrain_eval_summary.parent.mkdir(parents=True, exist_ok=True)
    paths.retrain_eval_summary.write_text("{}", encoding="utf-8")

    # Any unexpected call should fail the test.
    monkeypatch.setattr(runner, "build_initial_rule_pool", lambda **_: (_ for _ in ()).throw(AssertionError("called")))
    monkeypatch.setattr(runner, "build_initial_arm_pool", lambda **_: (_ for _ in ()).throw(AssertionError("called")))
    monkeypatch.setattr(
        runner,
        "ArmDrivenKGRefinementPipeline",
        type("P", (), {"from_paths": staticmethod(lambda **_: (_ for _ in ()).throw(AssertionError("called")))}),
    )
    monkeypatch.setattr(runner, "retrain_and_evaluate", lambda **_: (_ for _ in ()).throw(AssertionError("called")))

    runner.run_pipeline(
        run_dir=run_dir,
        model_dir=model_dir,
        target_relation="/r",
        dataset_dir=dataset_dir,
        target_triples=target_triples,
        candidate_triples=candidate_triples,
        force=False,
    )
