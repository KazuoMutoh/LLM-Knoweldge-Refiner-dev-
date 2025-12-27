import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simple_active_refine.amie import AmieRule, TriplePattern
from simple_active_refine.pipeline import (
    RefinedKG,
    RuleExtractionContext,
    TripleAcquisitionContext,
    TripleEvaluationContext,
    KGETrainingContext,
)
from simple_active_refine.pipeline_concrete import (
    AMIERuleExtractor,
    LLMWebTripleAcquirer,
    SimpleHeuristicTripleEvaluator,
    TrainRemovedTripleAcquirer,
)


def _dummy_rule():
    return AmieRule(
        head=TriplePattern("?x", "r", "?y"),
        body=[],
        support=None,
        std_conf=None,
        pca_conf=None,
        head_coverage=None,
        body_size=None,
        pca_body_size=None,
        raw="dummy",
    )


def test_train_removed_acquirer_and_evaluator(monkeypatch, tmp_path):
    # Prepare KG
    kg = RefinedKG(triples=[("h1", "r1", "t1")])

    # Prepare train_removed file with one new triple
    path = tmp_path / "train_removed.txt"
    path.write_text("h2\tr2\tt2\n", encoding="utf-8")

    # Rule extractor: patch AMIE to return a dummy rule
    dummy_rules = [_dummy_rule()]

    class _DummyAmieRules:
        def __init__(self, rules):
            self.rules = rules

    def _fake_run_amie(triples, min_pca=None, min_head_coverage=None):
        return _DummyAmieRules(dummy_rules)

    monkeypatch.setattr("simple_active_refine.pipeline_concrete.AmieRules.run_amie", _fake_run_amie)
    extractor = AMIERuleExtractor()
    extracted = extractor.extract(RuleExtractionContext(kg=kg))
    assert extracted.rules == dummy_rules

    # Acquisition
    acquirer = TrainRemovedTripleAcquirer()
    acquisition = acquirer.acquire(
        TripleAcquisitionContext(
            kg=kg, rules=extracted.rules, iteration=1, metadata={"train_removed_path": str(path)}
        )
    )
    assert acquisition.candidates_by_rule["train_removed"] == [("h2", "r2", "t2")]

    # Entity text for evaluator (accepts if entities exist)
    entity_text = tmp_path / "entity2textlong.txt"
    entity_text.write_text("h2\tH2 desc\nt2\tT2 desc\n", encoding="utf-8")
    evaluator = SimpleHeuristicTripleEvaluator()
    evaluation = evaluator.evaluate(
        TripleEvaluationContext(kg=kg, iteration=1, metadata={"entity_text_path": str(entity_text)}),
        acquisition,
    )
    assert evaluation.accepted_triples == [("h2", "r2", "t2")]
    assert evaluation.rule_rewards["train_removed"] > 0


def test_llm_web_triple_acquirer_without_retriever():
    acquirer = LLMWebTripleAcquirer(retriever=None)
    res = acquirer.acquire(TripleAcquisitionContext(kg=RefinedKG([]), rules=[], iteration=1))
    assert res.candidates_by_rule == {}
    assert res.diagnostics["n_candidates"] == 0


def test_pykeen_trainer_requires_dir_triples():
    PyKEENKGETrainer = pytest.importorskip("simple_active_refine.pipeline_concrete").PyKEENKGETrainer
    trainer = PyKEENKGETrainer(num_epochs=1)
    with pytest.raises(ValueError):
        trainer.train_and_evaluate(KGETrainingContext(kg=RefinedKG([]), output_dir=None, metadata={}))
