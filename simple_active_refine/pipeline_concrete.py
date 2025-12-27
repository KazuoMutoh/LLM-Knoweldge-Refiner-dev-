"""
Concrete strategy implementations for the rule-driven KG refinement pipeline.

The classes here implement the abstract interfaces defined in
`simple_active_refine.pipeline` while keeping the interfaces stable and
extension-friendly. Each class is intentionally thin and uses pluggable
callables so that heavier dependencies (AMIE+, LLM API calls, etc.) can be
swapped or mocked in tests.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from simple_active_refine.amie import AmieRule, AmieRules, TriplePattern
from simple_active_refine.knoweldge_retriever import LLMKnowledgeRetriever
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
    Triple,
    TripleAcquisitionContext,
    TripleAcquisitionResult,
    TripleEvaluationContext,
    TripleEvaluationResult,
)
from simple_active_refine.rule_history import RuleEvaluationRecord, RuleHistory
from simple_active_refine.util import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rule extractors
# ---------------------------------------------------------------------------


class AMIERuleExtractor(BaseRuleExtractor):
    """Rule extractor that runs AMIE+ and applies simple numeric filters."""

    def __init__(
        self,
        min_pca_conf: float = 0.01,
        min_head_coverage: float = 0.01,
        max_body_len: Optional[int] = None,
    ) -> None:
        self.min_pca_conf = min_pca_conf
        self.min_head_coverage = min_head_coverage
        self.max_body_len = max_body_len

    def extract(self, context: RuleExtractionContext) -> RuleExtractionResult:
        logger.info("Running AMIE+ rule extraction")

        triples = context.kg.triples
        rules = AmieRules.run_amie(
            triples,
            min_pca=self.min_pca_conf,
            min_head_coverage=self.min_head_coverage,
        )

        if self.max_body_len is not None:
            filtered = [r for r in rules.rules if len(r.body) <= self.max_body_len]
            rules = AmieRules(filtered)

        diagnostics = {
            "n_rules_raw": len(rules.rules),
        }
        return RuleExtractionResult(rules=rules.rules, diagnostics=diagnostics)


class AMIERuleExtractorWithLLMFilter(AMIERuleExtractor):
    """Rule extractor that post-filters AMIE+ rules via an injected LLM check.

    The LLM filter is provided as a callable taking an AmieRule and returning
    bool. This keeps the interface stable while allowing different LLM backends
    or mocking in tests.
    """

    def __init__(
        self,
        llm_filter: Optional[Callable[[AmieRule], bool]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.llm_filter = llm_filter

    def extract(self, context: RuleExtractionContext) -> RuleExtractionResult:
        base = super().extract(context)
        if not self.llm_filter:
            return base

        filtered_rules = [r for r in base.rules if self.llm_filter(r)]
        diagnostics = dict(base.diagnostics)
        diagnostics["n_rules_llm_pass"] = len(filtered_rules)
        return RuleExtractionResult(rules=filtered_rules, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Triple acquirers
# ---------------------------------------------------------------------------


class TrainRemovedTripleAcquirer(BaseTripleAcquirer):
    """Acquire triples from train_removed.txt (created by make_test_dataset.py).

    The path to train_removed.txt must be provided via
    `context.metadata["train_removed_path"]`. Triples already present in the KG
    are skipped.
    """

    def acquire(self, context: TripleAcquisitionContext) -> TripleAcquisitionResult:
        path = context.metadata.get("train_removed_path")
        if not path or not os.path.exists(path):
            logger.warning("train_removed_path not provided or missing; returning empty candidates")
            return TripleAcquisitionResult(candidates_by_rule={}, diagnostics={"n_candidates": 0})

        with open(path, "r", encoding="utf-8") as f:
            triples = [tuple(line.strip().split("\t")) for line in f if line.strip()]

        existing = set(context.kg.triples)
        deduped = [t for t in triples if t not in existing]
        diagnostics = {
            "n_loaded": len(triples),
            "n_deduped": len(deduped),
        }
        return TripleAcquisitionResult(candidates_by_rule={"train_removed": deduped}, diagnostics=diagnostics)


class LLMWebTripleAcquirer(BaseTripleAcquirer):
    """Acquire triples via Web/LLM using LLMKnowledgeRetriever.

    This is a thin wrapper to allow dependency injection and easy mocking.
    Provide a callable `retrieval_fn` if you need custom behavior; otherwise, a
    default LLMKnowledgeRetriever is used and called per rule with
    `retrieve_knowledge_for_triples` if available.
    """

    def __init__(
        self,
        retriever: Optional[LLMKnowledgeRetriever] = None,
        retrieval_fn: Optional[
            Callable[[LLMKnowledgeRetriever, AmieRule, Iterable[Triple]], List[Triple]]
        ] = None,
    ) -> None:
        self.retriever = retriever
        self.retrieval_fn = retrieval_fn

    def acquire(self, context: TripleAcquisitionContext) -> TripleAcquisitionResult:
        if self.retriever is None:
            logger.warning("LLMKnowledgeRetriever not provided; returning empty candidates")
            return TripleAcquisitionResult(candidates_by_rule={}, diagnostics={"n_candidates": 0})

        candidates_by_rule: Dict[str, List[Triple]] = {}
        total = 0

        for rule in context.rules:
            rule_id = rule.rule_id if hasattr(rule, "rule_id") else rule.to_string()
            if self.retrieval_fn:
                triples = self.retrieval_fn(self.retriever, rule, context.kg.triples)
            else:
                # Fallback: attempt to call a generic retrieval method if present.
                if hasattr(self.retriever, "retrieve_knowledge_for_triples"):
                    triples = self.retriever.retrieve_knowledge_for_triples([], rule)  # type: ignore
                else:
                    triples = []
            candidates_by_rule[rule_id] = list(triples)
            total += len(triples)

        diagnostics = {"n_candidates": total}
        return TripleAcquisitionResult(candidates_by_rule=candidates_by_rule, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Triple evaluator
# ---------------------------------------------------------------------------


class SimpleHeuristicTripleEvaluator(BaseTripleEvaluator):
    """Accept-or-reject triples with a lightweight heuristic.

    - Optionally load entity descriptions from metadata (key: entity_text_path)
      and require heads/tails to exist there.
    - Assign score 1.0 to accepted triples.
    - Rule rewards are proportional to the mean score of their triples.
    """

    def __init__(self) -> None:
        self.history = RuleHistory()

    def _load_entity_texts(self, path: Optional[str]) -> Dict[str, str]:
        if not path or not os.path.exists(path):
            return {}
        texts: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", maxsplit=1)
                if not parts:
                    continue
                ent = parts[0]
                desc = parts[1] if len(parts) > 1 else ""
                texts[ent] = desc
        return texts

    def evaluate(
        self,
        context: TripleEvaluationContext,
        acquisition: TripleAcquisitionResult,
    ) -> TripleEvaluationResult:
        entity_text_path = context.metadata.get("entity_text_path")
        entity_texts = self._load_entity_texts(entity_text_path)
        known_entities = set(entity_texts.keys()) if entity_texts else None

        accepted: List[Triple] = []
        rejected: List[Triple] = []
        triple_scores: Dict[Triple, float] = {}
        rule_rewards: Dict[str, float] = {}

        for rule_id, triples in acquisition.candidates_by_rule.items():
            scores: List[float] = []
            for triple in triples:
                h, _, t = triple
                if known_entities is not None and (h not in known_entities or t not in known_entities):
                    rejected.append(triple)
                    continue
                accepted.append(triple)
                triple_scores[triple] = 1.0
                scores.append(1.0)
            rule_rewards[rule_id] = float(sum(scores) / len(scores)) if scores else 0.0

        # Track history with a minimal record for downstream analysis
        for rule_id, reward in rule_rewards.items():
            placeholder_rule = AmieRule(
                head=TriplePattern("?s", "?p", "?o"),
                body=[],
                support=None,
                std_conf=None,
                pca_conf=None,
                head_coverage=None,
                body_size=None,
                pca_body_size=None,
                raw="placeholder",
            )
            rec = RuleEvaluationRecord(
                iteration=context.iteration,
                rule_id=rule_id,
                rule=placeholder_rule,
                target_triples=[],
                added_triples=accepted,
                score_changes=[reward] if reward else [],
                mean_score_change=reward,
                std_score_change=0.0,
                positive_changes=len(accepted) if reward else 0,
                negative_changes=0,
            )
            self.history.add_record(rec)

        diagnostics = {
            "n_candidates": sum(len(v) for v in acquisition.candidates_by_rule.values()),
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
        }
        return TripleEvaluationResult(
            accepted_triples=accepted,
            rejected_triples=rejected,
            rule_rewards=rule_rewards,
            triple_scores=triple_scores,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# KGE trainer
# ---------------------------------------------------------------------------


class PyKEENKGETrainer(BaseKGETrainer):
    """Train and evaluate a KGE model using PyKEEN wrappers."""

    def __init__(
        self,
        model_name: str = "TransE",
        num_epochs: int = 5,
        create_inverse_triples: bool = True,
        pipeline_kwargs: Optional[Dict] = None,
    ) -> None:
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.create_inverse_triples = create_inverse_triples
        self.pipeline_kwargs = pipeline_kwargs or {}

    def train_and_evaluate(self, context: KGETrainingContext) -> KGETrainingResult:
        from simple_active_refine.embedding import KnowledgeGraphEmbedding

        dir_triples = context.metadata.get("dir_triples")
        if not dir_triples:
            raise ValueError("dir_triples must be provided in metadata for KGE training")

        logger.info("Training KGE model with PyKEEN")
        kge = KnowledgeGraphEmbedding.train_model(
            model=self.model_name,
            dir_triples=dir_triples,
            create_inverse_triples=self.create_inverse_triples,
            dir_save=context.output_dir,
            num_epochs=self.num_epochs,
            **self.pipeline_kwargs,
        )
        metrics = kge.evaluate()

        return KGETrainingResult(
            hits_at_1=metrics.get("hits_at_1", 0.0),
            hits_at_3=metrics.get("hits_at_3", 0.0),
            hits_at_10=metrics.get("hits_at_10", 0.0),
            mrr=metrics.get("mean_reciprocal_rank", 0.0),
            model_path=context.output_dir,
            diagnostics={"num_epochs": self.num_epochs},
        )
