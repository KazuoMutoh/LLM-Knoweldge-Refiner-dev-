import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simple_active_refine.amie import AmieRule, TriplePattern
from simple_active_refine.pipeline import (
    BaseKGETrainer,
    BaseRuleExtractor,
    BaseTripleAcquirer,
    BaseTripleEvaluator,
    KGETrainingContext,
    KGETrainingResult,
    RefinedKG,
    RuleExtractionContext,
    RuleExtractionResult,
    TripleAcquisitionContext,
    TripleAcquisitionResult,
    TripleEvaluationContext,
    TripleEvaluationResult,
    RuleDrivenKGRefinementPipeline,
)
from main_v3 import RandomTripleAcquirer


class DummyRuleExtractor(BaseRuleExtractor):
    def __init__(self):
        self.rule = AmieRule(
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

    def extract(self, context: RuleExtractionContext) -> RuleExtractionResult:
        return RuleExtractionResult(rules=[self.rule], diagnostics={"n_rules": 1})


class DummyAcquirer(BaseTripleAcquirer):
    def __init__(self):
        self.triple = ("h", "r", "t")

    def acquire(self, context: TripleAcquisitionContext) -> TripleAcquisitionResult:
        return TripleAcquisitionResult(candidates_by_rule={"dummy": [self.triple]}, diagnostics={"n_candidates": 1})


class DummyEvaluator(BaseTripleEvaluator):
    def evaluate(self, context: TripleEvaluationContext, acquisition: TripleAcquisitionResult) -> TripleEvaluationResult:
        # Accept all candidates regardless of rule key for test simplicity
        triples = []
        for vals in acquisition.candidates_by_rule.values():
            triples.extend(vals)
        return TripleEvaluationResult(
            accepted_triples=triples,
            rejected_triples=[],
            rule_rewards={"dummy": 1.0},
            triple_scores={t: 1.0 for t in triples},
            diagnostics={"n_accepted": len(triples)},
        )


class DummyKGETrainer(BaseKGETrainer):
    def train_and_evaluate(self, context: KGETrainingContext) -> KGETrainingResult:
        return KGETrainingResult(
            hits_at_1=0.1,
            hits_at_3=0.2,
            hits_at_10=0.3,
            mrr=0.15,
            model_path=None,
            diagnostics={"dummy": True},
        )


def test_pipeline_end_to_end():
    initial = RefinedKG(triples=[("h0", "r0", "t0")])
    pipeline = RuleDrivenKGRefinementPipeline(
        rule_extractor=DummyRuleExtractor(),
        triple_acquirer=DummyAcquirer(),
        triple_evaluator=DummyEvaluator(),
        kge_trainer=DummyKGETrainer(),
    )

    result = pipeline.run(initial_kg=initial, num_rounds=2, kge_output_dir=None)

    # KG should include initial triple plus accepted ones from two rounds (no duplicates).
    assert ("h0", "r0", "t0") in result.final_kg.triples
    assert ("h", "r", "t") in result.final_kg.triples
    assert len(result.rounds) == 2
    # KGE result present
    assert result.kge_result is not None
    assert result.kge_result.hits_at_1 == 0.1


def test_pipeline_with_random_acquirer():
    initial = RefinedKG(triples=[("h0", "r0", "t0")])
    extractor = DummyRuleExtractor()
    evaluator = DummyEvaluator()
    trainer = DummyKGETrainer()

    # Use two distinct target triples to ensure sampling works without reuse
    target_triples = [("a", "r", "b"), ("c", "r", "d")]
    acquirer = RandomTripleAcquirer(target_triples=target_triples, n_targets_per_rule=1, reuse_targets=False)

    pipeline = RuleDrivenKGRefinementPipeline(
        rule_extractor=extractor,
        triple_acquirer=acquirer,
        triple_evaluator=evaluator,
        kge_trainer=trainer,
    )

    result = pipeline.run(initial_kg=initial, num_rounds=2, kge_output_dir=None)

    # Random acquirer should add at least one of the target triples across rounds
    assert any(triple in result.final_kg.triples for triple in target_triples)
    assert len(result.rounds) == 2
