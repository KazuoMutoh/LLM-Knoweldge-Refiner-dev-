"""
Pipeline interfaces and thin orchestration for rule-driven KG refinement.

This module defines stable, extensible interfaces for the four high-level steps:
1. Rule extraction from an existing KG.
2. Triple acquisition based on the extracted rules (internal + external).
3. Rule scoring / reward update (optional candidate filtering).
4. Post-refinement KGE training and evaluation.

Each step is represented by an abstract base class so concrete strategies can be
swapped or extended without changing the orchestration API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import statistics
from typing import Dict, Iterable, List, Optional, Tuple

from simple_active_refine.amie import AmieRule
from simple_active_refine.rule_history import RuleEvaluationRecord, RuleHistory
from simple_active_refine.rule_selector import RuleSelector, RuleWithId
from simple_active_refine.util import get_logger

logger = get_logger(__name__)

Triple = Tuple[str, str, str]


@dataclass
class RefinedKG:
    """A lightweight KG container for the refinement loop."""

    triples: List[Triple] = field(default_factory=list)

    def add_triples(self, new_triples: Iterable[Triple]) -> None:
        """Add unique triples to the KG snapshot."""
        existing = set(self.triples)
        for triple in new_triples:
            if triple not in existing:
                self.triples.append(triple)
                existing.add(triple)


@dataclass
class RuleExtractionContext:
    """Input payload for rule extraction."""

    kg: RefinedKG
    iteration: int = 0
    output_dir: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RuleExtractionResult:
    """Result of rule extraction."""

    rules: List[AmieRule]
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TripleAcquisitionContext:
    """Input payload for acquiring candidate triples."""

    kg: RefinedKG
    rules: List[AmieRule]
    iteration: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TripleAcquisitionResult:
    """Candidate triples grouped by rule."""

    candidates_by_rule: Dict[str, List[Triple]]
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TripleEvaluationContext:
    """Input payload for evaluating triples and updating rule scores."""

    kg: RefinedKG
    iteration: int
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TripleEvaluationResult:
    """Evaluation outcome for candidate triples and rules."""

    accepted_triples: List[Triple]
    rejected_triples: List[Triple] = field(default_factory=list)
    rule_rewards: Dict[str, float] = field(default_factory=dict)
    triple_scores: Dict[Triple, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass
class KGETrainingContext:
    """Input payload for post-refinement KGE training."""

    kg: RefinedKG
    output_dir: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class KGETrainingResult:
    """Outcome of the KGE training and evaluation."""

    hits_at_1: float
    hits_at_3: float
    hits_at_10: float
    mrr: float
    model_path: Optional[str] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineRoundResult:
    """Summary of a single refinement round."""

    iteration: int
    extracted_rules: List[AmieRule]
    acquisition: TripleAcquisitionResult
    evaluation: TripleEvaluationResult


@dataclass
class PipelineRunResult:
    """End-to-end run summary."""

    final_kg: RefinedKG
    rounds: List[PipelineRoundResult]
    kge_result: Optional[KGETrainingResult] = None


class BaseRuleExtractor(ABC):
    """Abstract rule extractor.

    Implementations may use AMIE+, heuristic filters, or LLM checks, but the
    interface remains stable.
    """

    @abstractmethod
    def extract(self, context: RuleExtractionContext) -> RuleExtractionResult:
        """Extract high-quality rules from the given KG context."""
        raise NotImplementedError


class BaseTripleAcquirer(ABC):
    """Abstract triple acquisition step.

    Implementations can perform internal rule-based link prediction, external
    Web/LLM retrieval, or hybrids.
    """

    @abstractmethod
    def acquire(self, context: TripleAcquisitionContext) -> TripleAcquisitionResult:
        """Acquire candidate triples given rules and the current KG."""
        raise NotImplementedError


class BaseTripleEvaluator(ABC):
    """Abstract triple evaluator and rule scorer.

    Note:
        In the simplest setup, candidate acquisition itself is treated as
        implicit acceptance, and this evaluator can be omitted. When provided,
        implementations may filter candidates (accept/reject) and/or assign
        richer scores/rewards.
    """

    @abstractmethod
    def evaluate(self, context: TripleEvaluationContext, acquisition: TripleAcquisitionResult) -> TripleEvaluationResult:
        """Score candidates, decide acceptance, and assign rewards to rules."""
        raise NotImplementedError


class BaseKGETrainer(ABC):
    """Abstract KGE training and evaluation step."""

    @abstractmethod
    def train_and_evaluate(self, context: KGETrainingContext) -> KGETrainingResult:
        """Train a KGE model on the refined KG and return evaluation metrics."""
        raise NotImplementedError


class RuleDrivenKGRefinementPipeline:
    """Orchestrates the four high-level steps with stable interfaces.

    The internal logic of each step is delegated to strategy objects so that
    implementations can evolve independently while the orchestration contract
    stays unchanged.
    """

    def __init__(
        self,
        rule_extractor: BaseRuleExtractor,
        triple_acquirer: BaseTripleAcquirer,
        triple_evaluator: Optional[BaseTripleEvaluator] = None,
        kge_trainer: Optional[BaseKGETrainer] = None,
        rule_selector: Optional[RuleSelector] = None,
        rule_history: Optional[RuleHistory] = None,
        n_select_rules: Optional[int] = None,
    ) -> None:
        self.rule_extractor = rule_extractor
        self.triple_acquirer = triple_acquirer
        self.triple_evaluator = triple_evaluator
        self.kge_trainer = kge_trainer
        self.rule_selector = rule_selector
        self.rule_history = rule_history
        self.n_select_rules = n_select_rules
        self._rule_pool: Dict[str, RuleWithId] = {}

    def run(
        self,
        initial_kg: RefinedKG,
        num_rounds: int,
        kge_output_dir: Optional[str] = None,
    ) -> PipelineRunResult:
        """Execute the refinement rounds, then optionally train/evaluate KGE."""
        current_kg = RefinedKG(triples=list(initial_kg.triples))
        rounds: List[PipelineRoundResult] = []

        for iteration in range(1, num_rounds + 1):
            logger.info(f"[Pipeline] Iteration {iteration}: extracting rules")
            extraction_ctx = RuleExtractionContext(kg=current_kg, iteration=iteration)
            extracted = self.rule_extractor.extract(extraction_ctx)

            # Map rules to stable IDs
            rule_pool_with_id: List[RuleWithId] = []
            for rule in extracted.rules:
                key = str(rule)
                if key not in self._rule_pool:
                    self._rule_pool[key] = RuleWithId.create(rule)
                else:
                    # refresh rule object in case attributes changed
                    self._rule_pool[key].rule = rule
                rule_pool_with_id.append(self._rule_pool[key])

            # Optional selection
            selected_with_id: List[RuleWithId]
            if self.rule_selector:
                k = self.n_select_rules or len(rule_pool_with_id)
                selected_with_id, _ = self.rule_selector.select_rules(rule_pool_with_id, k=k, iteration=iteration)
                selected_rules = [rwi.rule for rwi in selected_with_id]
                logger.info("[Pipeline] Iteration %d: selected %d/%d rules", iteration, len(selected_rules), len(rule_pool_with_id))
            else:
                selected_with_id = rule_pool_with_id
                selected_rules = extracted.rules

            logger.info(f"[Pipeline] Iteration {iteration}: acquiring triples")
            acquisition_ctx = TripleAcquisitionContext(
                kg=current_kg,
                rules=selected_rules,
                iteration=iteration,
            )
            acquisition = self.triple_acquirer.acquire(acquisition_ctx)

            # Candidate acquisition can be treated as implicit acceptance.
            if self.triple_evaluator is not None:
                logger.info(f"[Pipeline] Iteration {iteration}: evaluating triples")
                evaluation_ctx = TripleEvaluationContext(
                    kg=current_kg,
                    iteration=iteration,
                )
                evaluation = self.triple_evaluator.evaluate(evaluation_ctx, acquisition)
            else:
                accepted: List[Triple] = []
                rule_rewards: Dict[str, float] = {}
                for rule_key, triples in acquisition.candidates_by_rule.items():
                    accepted.extend(triples)
                    rule_rewards[rule_key] = float(len(triples))
                unique = list(dict.fromkeys([tuple(t) for t in accepted]))
                evaluation = TripleEvaluationResult(
                    accepted_triples=unique,
                    rejected_triples=[],
                    rule_rewards=rule_rewards,
                    triple_scores={t: 1.0 for t in unique},
                    diagnostics={"n_accepted": len(unique), "implicit_acceptance": 1.0},
                )

            # --- Rule evaluation bookkeeping (history/rewards) ---
            # Keep rule history aligned with the evaluator outputs, not the raw acquisition.
            if self.rule_history is not None:
                accepted_set = set(evaluation.accepted_triples)
                for rwi in selected_with_id:
                    rule_key = str(rwi.rule)
                    candidates = acquisition.candidates_by_rule.get(rule_key, [])
                    accepted_for_rule = [t for t in candidates if t in accepted_set]

                    reward = evaluation.rule_rewards.get(rule_key, float(len(accepted_for_rule)))
                    score_changes = [float(evaluation.triple_scores.get(t, 1.0)) for t in accepted_for_rule]
                    mean_score_change = float(statistics.fmean(score_changes)) if score_changes else 0.0
                    std_score_change = float(statistics.pstdev(score_changes)) if len(score_changes) > 1 else 0.0
                    positive_changes = len([s for s in score_changes if s > 0])
                    negative_changes = len([s for s in score_changes if s < 0])

                    record = RuleEvaluationRecord(
                        iteration=iteration,
                        rule_id=rwi.rule_id,
                        rule=rwi.rule,
                        target_triples=[],
                        added_triples=accepted_for_rule,
                        score_changes=score_changes,
                        mean_score_change=mean_score_change,
                        std_score_change=std_score_change,
                        positive_changes=positive_changes,
                        negative_changes=negative_changes,
                    )
                    self.rule_history.add_record(record)

            # --- KG update (apply accepted triples) ---
            current_kg.add_triples(evaluation.accepted_triples)

            rounds.append(
                PipelineRoundResult(
                    iteration=iteration,
                    extracted_rules=extracted.rules,
                    acquisition=acquisition,
                    evaluation=evaluation,
                )
            )

        kge_result: Optional[KGETrainingResult] = None
        if self.kge_trainer:
            logger.info("[Pipeline] Training KGE on refined KG")
            kge_ctx = KGETrainingContext(kg=current_kg, output_dir=kge_output_dir)
            kge_result = self.kge_trainer.train_and_evaluate(kge_ctx)

        return PipelineRunResult(final_kg=current_kg, rounds=rounds, kge_result=kge_result)
