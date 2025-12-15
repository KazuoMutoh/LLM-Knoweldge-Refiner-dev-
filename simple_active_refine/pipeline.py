"""
Pipeline interfaces and thin orchestration for rule-driven KG refinement.

This module defines stable, extensible interfaces for the four high-level steps:
1. Rule extraction from an existing KG.
2. Triple acquisition based on the extracted rules (internal + external).
3. Triple evaluation and rule reward update.
4. Post-refinement KGE training and evaluation.

Each step is represented by an abstract base class so concrete strategies can be
swapped or extended without changing the orchestration API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from simple_active_refine.amie import AmieRule
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
    """Abstract triple evaluator and rule scorer."""

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
        triple_evaluator: BaseTripleEvaluator,
        kge_trainer: Optional[BaseKGETrainer] = None,
    ) -> None:
        self.rule_extractor = rule_extractor
        self.triple_acquirer = triple_acquirer
        self.triple_evaluator = triple_evaluator
        self.kge_trainer = kge_trainer

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
            extraction_ctx = RuleExtractionContext(kg=current_kg)
            extracted = self.rule_extractor.extract(extraction_ctx)

            logger.info(f"[Pipeline] Iteration {iteration}: acquiring triples")
            acquisition_ctx = TripleAcquisitionContext(
                kg=current_kg,
                rules=extracted.rules,
                iteration=iteration,
            )
            acquisition = self.triple_acquirer.acquire(acquisition_ctx)

            logger.info(f"[Pipeline] Iteration {iteration}: evaluating triples")
            evaluation_ctx = TripleEvaluationContext(
                kg=current_kg,
                iteration=iteration,
            )
            evaluation = self.triple_evaluator.evaluate(evaluation_ctx, acquisition)

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
