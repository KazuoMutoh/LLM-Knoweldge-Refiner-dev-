"""Concrete triple acquirer implementations."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

from simple_active_refine.pipeline import BaseTripleAcquirer, TripleAcquisitionContext, TripleAcquisitionResult
from simple_active_refine.triples_editor import add_triples_for_single_rule
from simple_active_refine.knoweldge_retriever import (
    LLMKnowledgeRetriever,
    Entity as KR_Entity,
    Relation as KR_Relation,
)
from simple_active_refine.util import get_logger

logger = get_logger(__name__)

Triple = Tuple[str, str, str]


def _load_triples_from_tsv(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


class RuleBasedTripleAcquirer(BaseTripleAcquirer):
    """Use rules to select triples from a provided candidate pool (train_removed)."""

    def __init__(
        self,
        target_triples: List[Triple],
        candidate_dir: str,
        n_targets_per_rule: int,
        candidate_file: str = "train_removed.txt",
        reuse_targets: bool = False,
        dump_base_dir: Optional[str] = None,
    ) -> None:
        self.target_triples = target_triples
        self.candidate_dir = candidate_dir
        self.candidate_file = candidate_file
        self.n_targets_per_rule = n_targets_per_rule
        self.reuse_targets = reuse_targets
        self.dump_base_dir = dump_base_dir

    def acquire(self, context: TripleAcquisitionContext) -> TripleAcquisitionResult:
        used_targets = set()
        candidates_by_rule: Dict[str, List[Triple]] = {}
        f_candidate = os.path.join(self.candidate_dir, self.candidate_file)

        for rule in context.rules:
            available = self.target_triples if self.reuse_targets else [t for t in self.target_triples if t not in used_targets]
            if not available:
                logger.warning("[acq] No remaining target triples; skipping rule")
                continue

            sample_size = min(self.n_targets_per_rule, len(available))
            sampled = random.sample(available, sample_size)

            added_triples, _ = add_triples_for_single_rule(
                dir_triples=self.candidate_dir,
                rule=rule,
                target_triples=sampled,
                f_candidate_triples=f_candidate,
            )

            if added_triples:
                key = str(rule)
                candidates_by_rule[key] = [tuple(t) for t in added_triples]
            if not self.reuse_targets:
                used_targets.update(sampled)

        diagnostics = {
            "n_rules_with_candidates": len(candidates_by_rule),
            "n_total_candidates": sum(len(v) for v in candidates_by_rule.values()),
            "candidate_file": f_candidate,
        }
        if self.dump_base_dir:
            iter_dir = os.path.join(self.dump_base_dir, f"iter_{context.iteration}")
            os.makedirs(iter_dir, exist_ok=True)
            dump_path = os.path.join(iter_dir, "triple_acquirer_io.json")
            try:
                payload = {
                    "iteration": context.iteration,
                    "input": {
                        "n_targets_per_rule": self.n_targets_per_rule,
                        "candidate_dir": self.candidate_dir,
                        "candidate_file": self.candidate_file,
                        "rules": [str(r) for r in context.rules],
                    },
                    "output": {
                        "candidates_by_rule": {k: [list(t) for t in v] for k, v in candidates_by_rule.items()},
                        "diagnostics": diagnostics,
                    },
                }
                with open(dump_path, "w", encoding="utf-8") as fout:
                    json.dump(payload, fout, ensure_ascii=False, indent=2)
            except Exception as err:
                logger.warning("[acq] Failed to dump triple acquirer IO: %s", err)
        return TripleAcquisitionResult(candidates_by_rule=candidates_by_rule, diagnostics=diagnostics)


class RandomTripleAcquirer(BaseTripleAcquirer):
    """Ignore rules and sample triples randomly from a candidate pool."""

    def __init__(
        self,
        candidate_dir: Optional[str] = None,
        n_triples_per_rule: Optional[int] = None,
        candidate_file: str = "train_removed.txt",
        reuse_candidates: bool = False,
        reuse_targets: Optional[bool] = None,
        candidate_triples: Optional[List[Triple]] = None,
        target_triples: Optional[List[Triple]] = None,
        n_targets_per_rule: Optional[int] = None,
        dump_base_dir: Optional[str] = None,
    ) -> None:
        self.candidate_dir = candidate_dir
        self.candidate_file = candidate_file
        self.n_triples_per_rule = n_triples_per_rule or n_targets_per_rule or 0
        # Backward-compatible alias: some callers treat this as sampling "targets".
        if reuse_targets is not None:
            reuse_candidates = reuse_targets
        self.reuse_candidates = reuse_candidates
        self.dump_base_dir = dump_base_dir
        if candidate_triples is not None:
            self.candidate_triples = list(candidate_triples)
        elif candidate_dir:
            f_candidate = os.path.join(candidate_dir, candidate_file)
            self.candidate_triples = _load_triples_from_tsv(f_candidate) if os.path.exists(f_candidate) else []
        elif target_triples is not None:
            self.candidate_triples = list(target_triples)
        else:
            self.candidate_triples = []

    def acquire(self, context: TripleAcquisitionContext) -> TripleAcquisitionResult:
        remaining = list(self.candidate_triples)
        candidates_by_rule: Dict[str, List[Triple]] = {}

        for rule in context.rules:
            if not remaining:
                break

            sample_size = min(self.n_triples_per_rule, len(remaining))
            sampled = random.sample(remaining, sample_size)

            if not self.reuse_candidates:
                remaining = [t for t in remaining if t not in sampled]

            key = str(rule)
            candidates_by_rule[key] = [tuple(t) for t in sampled]

        diagnostics = {
            "n_rules_with_candidates": len(candidates_by_rule),
            "n_total_candidates": sum(len(v) for v in candidates_by_rule.values()),
            "reuse_candidates": self.reuse_candidates,
            "candidate_file": self.candidate_file,
        }
        if self.dump_base_dir:
            iter_dir = os.path.join(self.dump_base_dir, f"iter_{context.iteration}")
            os.makedirs(iter_dir, exist_ok=True)
            dump_path = os.path.join(iter_dir, "triple_acquirer_io.json")
            try:
                payload = {
                    "iteration": context.iteration,
                    "input": {
                        "n_triples_per_rule": self.n_triples_per_rule,
                        "reuse_candidates": self.reuse_candidates,
                        "candidate_dir": self.candidate_dir,
                        "candidate_file": self.candidate_file,
                        "rules": [str(r) for r in context.rules],
                    },
                    "output": {
                        "candidates_by_rule": {k: [list(t) for t in v] for k, v in candidates_by_rule.items()},
                        "diagnostics": diagnostics,
                    },
                }
                with open(dump_path, "w", encoding="utf-8") as fout:
                    json.dump(payload, fout, ensure_ascii=False, indent=2)
            except Exception as err:
                logger.warning("[acq] Failed to dump triple acquirer IO: %s", err)
        return TripleAcquisitionResult(candidates_by_rule=candidates_by_rule, diagnostics=diagnostics)


class WebSearchTripleAcquirer(BaseTripleAcquirer):
    """Fetch triples via LLM/web search per rule using target triples as seeds."""

    def __init__(
        self,
        target_triples: List[Triple],
        retriever: LLMKnowledgeRetriever,
        n_targets_per_rule: int,
        dump_base_dir: Optional[str] = None,
    ) -> None:
        self.target_triples = target_triples
        self.retriever = retriever
        self.n_targets_per_rule = n_targets_per_rule
        self.dump_base_dir = dump_base_dir

    def _to_relation(self, predicate: str) -> KR_Relation:
        return KR_Relation(id=predicate, label=predicate, description_short=predicate)

    def _to_entity(self, entity_id: str) -> KR_Entity:
        return KR_Entity(id=entity_id, label=entity_id, description_short=entity_id)

    def acquire(self, context: TripleAcquisitionContext) -> TripleAcquisitionResult:
        used_targets = set()
        candidates_by_rule: Dict[str, List[Triple]] = {}

        for rule in context.rules:
            available = [t for t in self.target_triples if t not in used_targets]
            if not available:
                break

            sample_size = min(self.n_targets_per_rule, len(available))
            sampled = random.sample(available, sample_size)

            collected: List[Triple] = []
            for head, rel, tail in sampled:
                entity = self._to_entity(head)
                relation = self._to_relation(rel)
                try:
                    rk = self.retriever.retrieve_knowledge_for_entity(entity, [relation])
                    for t in rk.triples:
                        collected.append((t.subject, t.predicate, t.object))
                except Exception as err:
                    logger.warning("[acq] Web retrieval failed for %s: %s", rule, err)

            if collected:
                key = str(rule)
                candidates_by_rule[key] = collected
                used_targets.update(sampled)

        diagnostics = {
            "n_rules_with_candidates": len(candidates_by_rule),
            "n_total_candidates": sum(len(v) for v in candidates_by_rule.values()),
            "mode": "web_search",
        }
        if self.dump_base_dir:
            iter_dir = os.path.join(self.dump_base_dir, f"iter_{context.iteration}")
            os.makedirs(iter_dir, exist_ok=True)
            dump_path = os.path.join(iter_dir, "triple_acquirer_io.json")
            try:
                payload = {
                    "iteration": context.iteration,
                    "input": {
                        "n_targets_per_rule": self.n_targets_per_rule,
                        "rules": [str(r) for r in context.rules],
                    },
                    "output": {
                        "candidates_by_rule": {k: [list(t) for t in v] for k, v in candidates_by_rule.items()},
                        "diagnostics": diagnostics,
                    },
                }
                with open(dump_path, "w", encoding="utf-8") as fout:
                    json.dump(payload, fout, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return TripleAcquisitionResult(candidates_by_rule=candidates_by_rule, diagnostics=diagnostics)
