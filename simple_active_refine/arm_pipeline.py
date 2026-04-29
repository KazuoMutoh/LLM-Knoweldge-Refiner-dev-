"""Arm-driven KG refinement pipeline.

This module orchestrates the v1 iterative refinement loop:
- select arms
- acquire evidence/body triples
- evaluate using proxy reward
- update KG (evidence-only)
- persist per-iteration outputs

It is intentionally minimal to allow experimentation without retraining KGE.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from collections import defaultdict

from simple_active_refine.amie import AmieRule, AmieRules
from simple_active_refine.arm import ArmWithId
from simple_active_refine.arm_builder import load_arm_pool_with_ids
from simple_active_refine.arm_history import ArmEvaluationRecord, ArmHistory
from simple_active_refine.arm_selector import ArmSelector, create_arm_selector
from simple_active_refine.arm_triple_acquirer_impl import LocalArmTripleAcquirer
from simple_active_refine.arm_triple_evaluator_impl import WitnessConflictArmEvaluator
from simple_active_refine.io_utils import get_iteration_dir, save_json, write_triples
from simple_active_refine.relation_priors import load_relation_priors
from simple_active_refine.triples_editor import TripleIndex, load_triples_tsv
from simple_active_refine.util import get_logger

logger = get_logger("arm_pipeline")

Triple = Tuple[str, str, str]


@dataclass
class ArmPipelineConfig:
    """Configuration for ArmDrivenKGRefinementPipeline."""

    base_output_path: str
    n_iter: int = 1
    k_sel: int = 3
    n_targets_per_arm: int = 50
    max_witness_per_head: Optional[int] = None
    selector_strategy: str = "ucb"
    selector_exploration_param: float = 1.0
    selector_epsilon: float = 0.1

    witness_weight: float = 1.0
    evidence_weight: float = 1.0

    # Optional: relation-level priors to weight witness counts (KGE-friendly weighting).
    # If not provided, will auto-detect <dir_triples>/relation_priors.json when present.
    relation_priors_path: Optional[str] = None
    # If True, do not use relation priors even if relation_priors_path is set
    # or <dir_triples>/relation_priors.json exists.
    disable_relation_priors: bool = False

    # Optional augmentation:
    # If accepted evidence introduces new entities not present in the current KG,
    # also add candidate triples (e.g., from train_removed.txt) that mention those
    # new entities. This helps avoid adding "dangling" entities that have only a
    # single evidence edge.
    add_incident_candidate_triples_for_new_entities: bool = True
    max_incident_candidate_triples_per_iteration: Optional[int] = None

    # Candidate source for evidence acquisition.
    # - local: use candidate_triples_path (e.g., train_removed.txt)
    # - web: build candidate triples via LLMKnowledgeRetriever per iteration
    candidate_source: str = "local"  # local|web

    # Web retrieval options (used when candidate_source == "web")
    web_llm_model: str = "gpt-4o"
    web_use_web_search: bool = True
    web_max_targets_total_per_iteration: int = 20
    web_max_triples_per_iteration: int = 200
    web_enable_entity_linking: bool = True  # Enable entity linking to match web entities with existing KG entities


class ArmDrivenKGRefinementPipeline:
    """Run iterative refinement using arms as decision units."""

    @staticmethod
    def _is_iteration_complete(iter_dir: Path) -> bool:
        """Return True if the iteration directory looks complete.

        We treat an iteration as complete if it has the key persisted artifacts
        required for reproducibility and resuming.
        """

        required = [
            iter_dir / "accepted_added_triples.tsv",
            iter_dir / "arm_history.pkl",
            iter_dir / "selected_arms.json",
        ]
        return all(p.exists() for p in required)

    @staticmethod
    def _iter_index(iter_dir: Path) -> int:
        return int(iter_dir.name.split("_")[1])

    def __init__(
        self,
        config: ArmPipelineConfig,
        arm_pool: List[ArmWithId],
        rule_pool: AmieRules,
        kg_train_triples: List[Triple],
        target_triples: List[Triple],
        candidate_triples: List[Triple],
        relation_texts: Optional[Dict[str, str]] = None,
        relation_priors: Optional[Dict[str, float]] = None,
        entity_texts: Optional[Dict[str, str]] = None,
        kg: Optional["TextAttributedKnoweldgeGraph"] = None,
        initial_kge_scores: Optional[Dict[Triple, float]] = None,
    ) -> None:
        self.config = config
        self.arm_pool = arm_pool
        self.rule_pool = rule_pool
        self.kg_set = set(kg_train_triples)
        self.target_triples = list(target_triples)
        self.candidate_triples = list(candidate_triples)
        self.relation_texts: Dict[str, str] = dict(relation_texts or {})
        self.relation_priors: Dict[str, float] = dict(relation_priors or {})
        self.entity_texts: Dict[str, str] = dict(entity_texts or {})
        self.kg = kg  # TextAttributedKnoweldgeGraph for entity linking
        # KGE scores for target triples from the previous KGE training iteration.
        # When provided, the new (1 - s_prev) * novelty-witness reward formula is used.
        self.initial_kge_scores: Optional[Dict[Triple, float]] = dict(initial_kge_scores) if initial_kge_scores else None

        # Build a cheap incident index for candidate triples so we can quickly
        # retrieve triples mentioning newly introduced entities.
        self._candidate_triples_by_entity: DefaultDict[str, List[Triple]] = defaultdict(list)
        for s, p, o in self.candidate_triples:
            self._candidate_triples_by_entity[s].append((s, p, o))
            self._candidate_triples_by_entity[o].append((s, p, o))

        self.history = ArmHistory()

        # Semantic context: target predicate(s) (usually one relation like nationality).
        target_predicates = sorted({t[1] for t in self.target_triples})
        selector_kwargs: Dict = {}
        if config.selector_strategy == "ucb":
            selector_kwargs["exploration_param"] = config.selector_exploration_param
        elif config.selector_strategy == "epsilon_greedy":
            selector_kwargs["epsilon"] = config.selector_epsilon

        self.selector = create_arm_selector(
            strategy=config.selector_strategy,
            history=self.history,
            target_predicates=target_predicates,
            relation_texts=self.relation_texts,
            entity_texts=self.entity_texts,
            **selector_kwargs,
        )

        self.acquirer = LocalArmTripleAcquirer(
            n_targets_per_arm=config.n_targets_per_arm,
            max_witness_per_head=config.max_witness_per_head,
            relation_priors=self.relation_priors,
            random_seed=0,
        )
        self.evaluator = WitnessConflictArmEvaluator(
            witness_weight=config.witness_weight,
            evidence_weight=config.evidence_weight,
            hypothesis_predicates=sorted({t[1] for t in self.target_triples}),
        )

        self._hypothesis_predicates: Set[str] = set(sorted({t[1] for t in self.target_triples}))

        self.rule_by_key: Dict[str, AmieRule] = {str(r): r for r in self.rule_pool.rules}

        # Annotate arm metadata with derived predicates so the selector/LLM can reason semantically.
        for awi in self.arm_pool:
            md = awi.arm.metadata if isinstance(awi.arm.metadata, dict) else {}
            body_preds: Set[str] = set()
            head_preds: Set[str] = set()
            for rule_key in awi.arm.rule_keys:
                rule = self.rule_by_key.get(rule_key)
                if rule is None:
                    continue
                head_preds.add(rule.head.p)
                for tp in rule.body:
                    body_preds.add(tp.p)

            md.setdefault("body_predicates", sorted(body_preds))
            md.setdefault("head_predicates", sorted(head_preds))
            md.setdefault(
                "body_predicate_texts",
                {p: self.relation_texts.get(p, "") for p in sorted(body_preds) if self.relation_texts.get(p)},
            )
            md.setdefault(
                "head_predicate_texts",
                {p: self.relation_texts.get(p, "") for p in sorted(head_preds) if self.relation_texts.get(p)},
            )
            awi.arm.metadata = md

    @staticmethod
    def _entities_in_triples(triples: Set[Triple] | List[Triple]) -> Set[str]:
        entities: Set[str] = set()
        for s, _, o in triples:
            entities.add(s)
            entities.add(o)
        return entities

    @staticmethod
    def from_paths(
        config: ArmPipelineConfig,
        initial_arms_path: str,
        rule_pool_pkl: str,
        dir_triples: str,
        target_triples_path: str,
        candidate_triples_path: str,
    ) -> "ArmDrivenKGRefinementPipeline":
        """Create pipeline from file paths.

        Args:
            config: Pipeline config.
            initial_arms_path: Path to initial_arms.json or initial_arms.pkl.
            rule_pool_pkl: Path to initial_rule_pool.pkl (AmieRules).
            dir_triples: Directory with train.txt.
            target_triples_path: Path to target_triples.txt.
            candidate_triples_path: Path to candidate triples (e.g., train_removed.txt).

        Returns:
            ArmDrivenKGRefinementPipeline
        """
        
        # Backup original dataset before any modifications (if not already backed up)
        import shutil
        dataset_path = Path(dir_triples)
        backup_path = dataset_path.parent / f"{dataset_path.name}_original_backup"
        
        if not backup_path.exists():
            logger.info(f"Creating backup of original dataset: {backup_path}")
            shutil.copytree(dataset_path, backup_path)
            logger.info(f"Backup created successfully")
        else:
            logger.info(f"Backup already exists at: {backup_path}")

        arm_pool = load_arm_pool_with_ids(initial_arms_path)
        rule_pool = AmieRules.from_pickle(rule_pool_pkl)

        kg_train_triples = load_triples_tsv(str(Path(dir_triples) / "train.txt"))
        target_triples = load_triples_tsv(target_triples_path)
        candidate_triples = load_triples_tsv(candidate_triples_path)

        relation_priors: Dict[str, float] = {}
        if not config.disable_relation_priors:
            pri_path: Optional[Path] = None
            if config.relation_priors_path:
                pri_path = Path(config.relation_priors_path)
            else:
                auto = Path(dir_triples) / "relation_priors.json"
                if auto.exists():
                    pri_path = auto
            if pri_path is not None and pri_path.exists():
                relation_priors = load_relation_priors(pri_path)

        # Optional: load relation descriptions if available.
        relation_texts: Dict[str, str] = {}
        rel_path = Path(dir_triples) / "relation2text.txt"
        if rel_path.exists():
            with open(rel_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    relation, text = parts[0].strip(), parts[1].strip()
                    if relation and text:
                        relation_texts[relation] = text

        # Optional: load entity labels/descriptions if available.
        # Prefer entity2textlong.txt when present (richer descriptions), else entity2text.txt.
        entity_texts: Dict[str, str] = {}
        ent_paths = [Path(dir_triples) / "entity2textlong.txt", Path(dir_triples) / "entity2text.txt"]
        ent_path = next((p for p in ent_paths if p.exists()), None)
        if ent_path is not None:
            with open(ent_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue
                    entity_id, text = parts[0].strip(), parts[1].strip()
                    if entity_id and text:
                        entity_texts[entity_id] = text

        # Initialize TextAttributedKnoweldgeGraph for entity linking (if web source is enabled)
        kg = None
        if config.candidate_source == "web" and config.web_enable_entity_linking:
            from simple_active_refine.knoweldge_retriever import TextAttributedKnoweldgeGraph
            try:
                kg = TextAttributedKnoweldgeGraph(dir_triples=dir_triples)
                logger.info("TextAttributedKnoweldgeGraph initialized for entity linking")
            except Exception as e:
                logger.warning("Failed to initialize TextAttributedKnoweldgeGraph for entity linking: %s", e)
                kg = None

        return ArmDrivenKGRefinementPipeline(
            config=config,
            arm_pool=arm_pool,
            rule_pool=rule_pool,
            kg_train_triples=kg_train_triples,
            target_triples=target_triples,
            candidate_triples=candidate_triples,
            relation_texts=relation_texts,
            relation_priors=relation_priors,
            entity_texts=entity_texts,
            kg=kg,
        )

    def run(self) -> None:
        """Run iterative refinement and write per-iteration outputs."""

        base = Path(self.config.base_output_path)
        base.mkdir(parents=True, exist_ok=True)

        # Resume support: if prior iterations exist under base_output_path,
        # rebuild in-memory state (kg_set + history) and continue from the
        # next iteration.
        prior_iter_dirs = [p for p in base.glob("iter_*") if p.is_dir()]
        prior_iter_dirs = sorted(prior_iter_dirs, key=self._iter_index)
        completed_iter_dirs = [p for p in prior_iter_dirs if self._is_iteration_complete(p)]

        start_iteration = 1
        if completed_iter_dirs:
            last_completed_dir = completed_iter_dirs[-1]
            last_completed = self._iter_index(last_completed_dir)

            # Rebuild KG state by replaying accepted additions.
            for it_dir in completed_iter_dirs:
                try:
                    accepted_added = load_triples_tsv(str(it_dir / "accepted_added_triples.tsv"))
                except Exception as e:
                    logger.warning("Failed to load accepted_added_triples for resume from %s: %s", it_dir, e)
                    continue
                for t in accepted_added:
                    self.kg_set.add(t)

            # Load ArmHistory into the existing history object so the selector
            # keeps referencing the same instance.
            try:
                loaded_history = ArmHistory.load(str(last_completed_dir / "arm_history.pkl"))
                self.history.__dict__.update(loaded_history.__dict__)
            except Exception as e:
                logger.warning("Failed to load ArmHistory for resume from %s: %s", last_completed_dir, e)

            start_iteration = last_completed + 1
            logger.info(
                "Resume detected: completed up to iter_%d under %s; continuing from iter_%d",
                last_completed,
                base,
                start_iteration,
            )

        if start_iteration > self.config.n_iter:
            logger.info(
                "Nothing to do: start_iteration=%d exceeds n_iter=%d (base_output_path=%s)",
                start_iteration,
                self.config.n_iter,
                base,
            )
            return

        for iteration in range(start_iteration, self.config.n_iter + 1):
            iter_dir = get_iteration_dir(base, iteration)
            iter_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Iteration %d/%d: selecting arms (k_sel=%d)", iteration, self.config.n_iter, self.config.k_sel)
            selected_arms, policy_text = self.selector.select_arms(
                self.arm_pool,
                k=self.config.k_sel,
                iteration=iteration,
            )

            if self.config.candidate_source not in {"local", "web"}:
                raise ValueError(f"Unsupported candidate_source: {self.config.candidate_source}")

            # Sample targets deterministically in the pipeline so local and web modes
            # can share the same targets when needed.
            rng = random.Random(self.acquirer.random_seed + int(iteration))
            targets_all = list(self.target_triples)
            n = min(self.config.n_targets_per_arm, len(targets_all))
            targets_by_arm: Dict[str, List[Triple]] = {}
            for awi in selected_arms:
                targets_by_arm[awi.arm_id] = rng.sample(targets_all, n) if n > 0 else []

            web_candidates: List[Triple] = []
            web_provenance: Dict[str, Dict] = {}
            web_entities: Dict[str, Dict] = {}
            if self.config.candidate_source == "web":
                web_candidates, web_provenance, web_entities = self._retrieve_web_candidates(
                    selected_arms=selected_arms,
                    targets_by_arm=targets_by_arm,
                    iteration=iteration,
                )
                # Persist raw web candidates + provenance + entities for audit/repro.
                write_triples(iter_dir / "web_retrieved_triples.tsv", web_candidates)
                save_json(iter_dir / "web_provenance.json", web_provenance)
                save_json(iter_dir / "web_entities.json", web_entities)
                
                # Add web-retrieved entities to KG (for next iteration's TAKG)
                if self.kg and web_entities:
                    from simple_active_refine.knoweldge_retriever import Entity as KR_Entity
                    web_entity_list = []
                    for entity_id, entity_info in web_entities.items():
                        web_entity = KR_Entity(
                            id=entity_id,
                            label=entity_info.get("label", entity_id),
                            description_short=entity_info.get("description_short", ""),
                            description=entity_info.get("description", ""),
                            source=entity_info.get("source", "")
                        )
                        web_entity_list.append(web_entity)
                    self.kg.add_entities(web_entity_list)
                    logger.info("[web] Added %d web entities to KG (TAKG)", len(web_entity_list))

            # Build index once per iteration.
            # - local: use current KG + candidate pool (train_removed, etc.)
            # - web: use current KG + web-retrieved candidates
            if self.config.candidate_source == "local":
                current_candidate_set = set(self.candidate_triples)
                idx = TripleIndex(list(self.kg_set) + self.candidate_triples)
            else:
                current_candidate_set = set(web_candidates)
                idx = TripleIndex(list(self.kg_set) + web_candidates)

            acquisition = self.acquirer.acquire(
                selected_arms=selected_arms,
                target_triples=self.target_triples,
                candidates=idx,
                rule_by_key=self.rule_by_key,
                iteration=iteration,
                provided_targets_by_arm=targets_by_arm,
                candidate_set=current_candidate_set if self.initial_kge_scores is not None else None,
            )

            evaluation = self.evaluator.evaluate(
                acquisition=acquisition,
                current_kg_triples=self.kg_set,
                prev_kge_scores=self.initial_kge_scores,
            )

            # Compute extra candidate triples incident to any newly introduced entities.
            entities_before = self._entities_in_triples(self.kg_set)
            entities_in_evidence = self._entities_in_triples(evaluation.accepted_evidence_triples)
            new_entities = entities_in_evidence - entities_before

            accepted_incident_set: Set[Triple] = set()
            if self.config.add_incident_candidate_triples_for_new_entities and new_entities:
                for ent in sorted(new_entities):
                    for tr in self._candidate_triples_by_entity.get(ent, []):
                        if tr in self.kg_set:
                            continue
                        if tr[1] in self._hypothesis_predicates:
                            continue
                        accepted_incident_set.add(tr)

                if self.config.max_incident_candidate_triples_per_iteration is not None:
                    k = int(self.config.max_incident_candidate_triples_per_iteration)
                    if k >= 0 and len(accepted_incident_set) > k:
                        accepted_incident_set = set(sorted(accepted_incident_set)[:k])

            accepted_added_triples = sorted(set(evaluation.accepted_evidence_triples) | accepted_incident_set)

            # Update history per selected arm.
            for awi in selected_arms:
                arm_id = awi.arm_id

                targets = acquisition.targets_by_arm.get(arm_id, [])
                witness_by_target = acquisition.witness_by_arm_and_target.get(arm_id, {})
                witness_sum = float(sum(witness_by_target.values()))

                weighted_map = getattr(acquisition, "witness_score_by_arm_and_target", None) or {}
                witness_score_by_target = weighted_map.get(arm_id, {})
                witness_score_sum = float(sum(witness_score_by_target.values())) if witness_score_by_target else 0.0
                targets_total = len(targets)
                targets_with_witness = sum(1 for w in witness_by_target.values() if w > 0)

                evidence_all = acquisition.evidence_by_arm.get(arm_id, [])
                evidence_new = [t for t in evidence_all if t not in self.kg_set]
                evidence_existing = len(evidence_all) - len(evidence_new)

                target_entities: Set[str] = set()
                for s, _, o in targets:
                    target_entities.add(s)
                    target_entities.add(o)
                overlap_new = 0
                for s, _, o in evidence_new:
                    if s in target_entities or o in target_entities:
                        overlap_new += 1
                overlap_rate_new = (float(overlap_new) / float(len(evidence_new))) if evidence_new else 0.0

                novelty_map_arm = (
                    (getattr(acquisition, "novelty_witness_by_arm_and_target", None) or {}).get(arm_id, {})
                )
                novelty_witness_sum = float(sum(novelty_map_arm.values()))

                record = ArmEvaluationRecord(
                    iteration=iteration,
                    arm_id=arm_id,
                    arm=awi.arm,
                    target_triples=targets,
                    added_triples=evidence_new,
                    reward=float(evaluation.reward_by_arm.get(arm_id, 0.0)),
                    diagnostics={
                        "targets_total": float(targets_total),
                        "targets_with_witness": float(targets_with_witness),
                        "target_coverage": (float(targets_with_witness) / float(targets_total)) if targets_total else 0.0,
                        "witness_sum": witness_sum,
                        "mean_witness_per_target": (witness_sum / float(targets_total)) if targets_total else 0.0,
                        "witness_score_sum": float(witness_score_sum),
                        "mean_witness_score_per_target": (float(witness_score_sum) / float(targets_total)) if targets_total else 0.0,
                        "novelty_witness_sum": novelty_witness_sum,
                        "mean_novelty_witness_per_target": (novelty_witness_sum / float(targets_total)) if targets_total else 0.0,
                        "evidence_total": float(len(evidence_all)),
                        "evidence_new": float(len(evidence_new)),
                        "evidence_existing": float(evidence_existing),
                        "evidence_new_overlap_rate_with_targets": float(overlap_rate_new),
                    },
                    evidence_triples=evidence_all,
                    witness_by_target=witness_by_target,
                )
                self.history.add_record(record)

            # Update KG: accepted evidence plus optional incident candidate triples.
            for t in accepted_added_triples:
                self.kg_set.add(t)
            
            # Add accepted triples to TextAttributedKnoweldgeGraph (for next iteration's TAKG)
            if self.kg and accepted_added_triples:
                from simple_active_refine.knoweldge_retriever import Triple as KR_Triple
                kg_triple_list = [
                    KR_Triple(subject=s, predicate=p, object=o)
                    for s, p, o in accepted_added_triples
                ]
                self.kg.add_triples(kg_triple_list, data_type='train')
                logger.info("[kg] Added %d accepted triples to KG (TAKG)", len(kg_triple_list))

            # Persist outputs.
            selected_json = {
                "iteration": iteration,
                "k_sel": self.config.k_sel,
                "selector_strategy": self.config.selector_strategy,
                "policy_text": policy_text,
                "selected_arms": [
                    {
                        "arm_id": a.arm_id,
                        "arm_type": a.arm.arm_type,
                        "rule_keys": list(a.arm.rule_keys),
                        "metadata": dict(a.arm.metadata),
                        "reward": float(evaluation.reward_by_arm.get(a.arm_id, 0.0)),
                        "diagnostics": self.history.get_records_for_arm(a.arm_id)[-1].diagnostics
                        if self.history.get_records_for_arm(a.arm_id)
                        else {},
                    }
                    for a in selected_arms
                ],
            }

            save_json(iter_dir / "selected_arms.json", selected_json)
            write_triples(iter_dir / "accepted_evidence_triples.tsv", evaluation.accepted_evidence_triples)
            write_triples(iter_dir / "accepted_incident_triples.tsv", sorted(accepted_incident_set))
            write_triples(iter_dir / "accepted_added_triples.tsv", accepted_added_triples)
            write_triples(iter_dir / "pending_hypothesis_triples.tsv", evaluation.pending_hypothesis_triples)
            self.history.save(str(iter_dir / "arm_history.pkl"))
            self.history.save_json(str(iter_dir / "arm_history.json"))

            diagnostics = dict(evaluation.diagnostics)
            diagnostics.update(
                {
                    "accepted_evidence_total": float(len(evaluation.accepted_evidence_triples)),
                    "accepted_incident_total": float(len(accepted_incident_set)),
                    "accepted_added_total": float(len(accepted_added_triples)),
                    "new_entities_in_evidence_total": float(len(new_entities)),
                }
            )
            save_json(iter_dir / "diagnostics.json", diagnostics)
            
            # Save TextAttributedKnoweldgeGraph to files (entity2text.txt, entity2textlong.txt, train.txt)
            if self.kg:
                self.kg.save_to_files()
                logger.info("[kg] Saved TextAttributedKnoweldgeGraph to files for iteration %d", iteration)

            logger.info(
                "Iteration %d done: added=%d pending=%d",
                iteration,
                len(accepted_added_triples),
                len(evaluation.pending_hypothesis_triples),
            )

    def _retrieve_web_candidates(
        self,
        *,
        selected_arms: List[ArmWithId],
        targets_by_arm: Dict[str, List[Triple]],
        iteration: int,
    ) -> tuple[List[Triple], Dict[str, Dict], Dict[str, Dict]]:
        """Retrieve candidate triples from the web via LLMKnowledgeRetriever.

        Notes:
        - This is best-effort: failures should not abort the iteration.
        - Retrieved triples with hypothesis predicates (target relations) are filtered out.
        - New entities are normalized to stable IDs based on (label, source_url) when available.
        - Entity linking is performed to match web entities with existing KG entities.
        """

        from simple_active_refine.knoweldge_retriever import (
            LLMKnowledgeRetriever,
            Entity as KR_Entity,
            Relation as KR_Relation,
            KnowledgeRefiner,
        )

        # Collect body predicates with directionality per arm.
        # Direction is defined relative to the rule head subject variable.
        # - head: (head_var, pred, ?)
        # - tail: (?, pred, head_var)
        pred_pos_by_arm: Dict[str, List[tuple[str, str]]] = {}
        rules_by_arm: Dict[str, List[AmieRule]] = {}
        for awi in selected_arms:
            pairs_for_arm: Set[tuple[str, str]] = set()
            rules_for_arm: List[AmieRule] = []
            for rule_key in awi.arm.rule_keys:
                rule = self.rule_by_key.get(rule_key)
                if rule is None:
                    continue
                rules_for_arm.append(rule)
                head_var = rule.head.s
                for tp in rule.body:
                    if tp.s == head_var:
                        pairs_for_arm.add((tp.p, "head"))
                    if tp.o == head_var:
                        pairs_for_arm.add((tp.p, "tail"))

            if not pairs_for_arm:
                md = awi.arm.metadata if isinstance(awi.arm.metadata, dict) else {}
                preds = md.get("body_predicates")
                if isinstance(preds, list):
                    for p in preds:
                        if p:
                            pairs_for_arm.add((str(p), "head"))

            pred_pos_by_arm[awi.arm_id] = sorted(pairs_for_arm)
            rules_by_arm[awi.arm_id] = rules_for_arm

        # Flatten (arm_id, target_triple) pairs and cap total web calls.
        pairs: List[tuple[str, Triple]] = []
        for arm_id, ts in targets_by_arm.items():
            for t in ts:
                pairs.append((arm_id, t))

        rng = random.Random(1000 + int(iteration))
        if self.config.web_max_targets_total_per_iteration is not None:
            k = int(self.config.web_max_targets_total_per_iteration)
            if k >= 0 and len(pairs) > k:
                pairs = rng.sample(pairs, k)

        retriever = LLMKnowledgeRetriever(
            kg=None,
            llm_model=self.config.web_llm_model,
            use_web_search=bool(self.config.web_use_web_search),
        )
        
        # Initialize KnowledgeRefiner for entity linking (if enabled)
        refiner = None
        if self.config.web_enable_entity_linking and self.kg:
            from simple_active_refine.knoweldge_retriever import KnowledgeRefiner
            refiner = KnowledgeRefiner(kg=self.kg)
            logger.info("[web] Entity linking enabled")

        out_triples: List[Triple] = []
        provenance: Dict[str, Dict] = {}
        seen: Set[Triple] = set()
        web_entities_dict: Dict[str, Dict] = {}  # web_entity_id -> entity info
        entity_link_map: Dict[str, str] = {}  # web_entity_id -> kg_entity_id

        def _stable_id(label: str, source_url: str) -> str:
            s = f"{label}|{source_url}".encode("utf-8")
            return "web:" + hashlib.sha1(s).hexdigest()[:16]

        def _ingest_retrieved(
            *,
            rk: "RetrievedKnowledge",
            arm_id: str,
        ) -> tuple[List[Triple], bool]:
            """Ingest retrieved knowledge into the candidate pool.

            Returns:
                (added_triples, cap_reached)
            """
            added: List[Triple] = []

            # Build stable ID mapping for newly created entities (e1,e2,...) using returned entities.
            id_map: Dict[str, str] = {}
            for e in getattr(rk, "entities", []) or []:
                if not getattr(e, "id", ""):
                    continue
                if not str(e.id).startswith("e"):
                    continue
                label = str(getattr(e, "description_short", "") or getattr(e, "label", "") or e.id)
                url = str(getattr(e, "source", "") or "")
                if label and url:
                    web_id = _stable_id(label, url)
                    id_map[str(e.id)] = web_id

                    # Store entity information in web_entities_dict
                    if web_id not in web_entities_dict:
                        web_entities_dict[web_id] = {
                            "label": label,
                            "description_short": str(getattr(e, "description_short", "") or ""),
                            "description": str(getattr(e, "description", "") or ""),
                            "source": url,
                            "iteration": int(iteration),
                            "arm_id": arm_id,
                        }

                    # Perform entity linking if enabled and not already linked
                    if refiner and web_id not in entity_link_map:
                        try:
                            same_entities = refiner.find_same_entity(e, top_k=5, similarity_threshold=0.7)
                            if same_entities:
                                matched_entity = same_entities[0]
                                entity_link_map[web_id] = matched_entity.id
                                web_entities_dict[web_id]["linked_to"] = matched_entity.id
                                logger.info("[web] Entity linking: %s -> %s (%s)", web_id, matched_entity.id, label)
                        except Exception as link_err:
                            logger.warning("[web] Entity linking failed for %s: %s", web_id, link_err)
                else:
                    # Fallback: iteration-scoped unique-ish id
                    id_map[str(e.id)] = f"web:iter{iteration}:{arm_id}:{e.id}"

            for t in getattr(rk, "triples", []) or []:
                ss = id_map.get(str(t.subject), str(t.subject))
                pp = str(t.predicate)
                oo = id_map.get(str(t.object), str(t.object))

                # Apply entity linking mapping
                ss = entity_link_map.get(ss, ss)
                oo = entity_link_map.get(oo, oo)

                # Do not inject hypothesis predicates into candidate pool.
                if pp in self._hypothesis_predicates:
                    continue

                tr = (ss, pp, oo)
                if tr in self.kg_set or tr in seen:
                    continue

                seen.add(tr)
                out_triples.append(tr)
                added.append(tr)

                key = "\t".join(tr)
                prov = {"source": "web", "iteration": int(iteration), "arm_id": arm_id}
                src = getattr(t, "source", None)
                if src:
                    prov["url"] = str(src)
                provenance[key] = prov

                if self.config.web_max_triples_per_iteration is not None:
                    cap = int(self.config.web_max_triples_per_iteration)
                    if cap >= 0 and len(out_triples) >= cap:
                        return added, True

            return added, False

        for arm_id, (s, p, o) in pairs:
            pred_pos_pairs = pred_pos_by_arm.get(arm_id, [])
            if not pred_pos_pairs:
                continue

            def _do_retrieve(entity_id: str, rel_pos_pairs: List[tuple[str, str]]) -> tuple[List[Triple], bool]:
                if not rel_pos_pairs:
                    return [], False

                # Cap relations per call to keep web calls bounded.
                rel_pos_pairs = rel_pos_pairs[:5]
                relations: List[KR_Relation] = []
                for pred, pos in rel_pos_pairs:
                    rel_text = self.relation_texts.get(pred, "")
                    relations.append(
                        KR_Relation(
                            id=pred,
                            label=pred,
                            description_short=rel_text or pred,
                            position=pos,
                        )
                    )

                entity = KR_Entity(id=entity_id, label=entity_id, description_short=entity_id)
                try:
                    rk_local = retriever.retrieve_knowledge_for_entity(entity, relations)
                except Exception as err:
                    logger.warning("[web] retrieve failed arm=%s entity=%s: %s", arm_id, entity_id, err)
                    return [], False

                return _ingest_retrieved(rk=rk_local, arm_id=arm_id)

            added_1hop, cap_reached = _do_retrieve(s, pred_pos_pairs)
            if cap_reached:
                return out_triples, provenance, web_entities_dict

            # Lightweight 2-hop expansion for multi-hop rules:
            # If a rule body introduces intermediate variables, query those
            # discovered intermediate entities for the remaining body predicates.
            rules = rules_by_arm.get(arm_id, [])
            rule = rules[0] if rules else None
            if rule is None:
                continue

            head_var = rule.head.s
            # Build bindings for variables directly connected to head_var.
            var_to_entities: DefaultDict[str, Set[str]] = defaultdict(set)
            for tp in rule.body:
                if tp.s == head_var:
                    for (ss, pp, oo) in added_1hop:
                        if ss == s and pp == tp.p:
                            var_to_entities[tp.o].add(oo)
                elif tp.o == head_var:
                    for (ss, pp, oo) in added_1hop:
                        if oo == s and pp == tp.p:
                            var_to_entities[tp.s].add(ss)

            # Build 2-hop queries from any known intermediate bindings.
            second_hop_queries: List[tuple[str, List[tuple[str, str]]]] = []
            for tp in rule.body:
                if tp.s == head_var or tp.o == head_var:
                    continue

                # If subject var is bound, retrieve outgoing.
                if tp.s in var_to_entities:
                    for ent in list(sorted(var_to_entities[tp.s]))[:2]:
                        second_hop_queries.append((ent, [(tp.p, "head")]))

                # If object var is bound, retrieve incoming.
                if tp.o in var_to_entities:
                    for ent in list(sorted(var_to_entities[tp.o]))[:2]:
                        second_hop_queries.append((ent, [(tp.p, "tail")]))

            # De-dup and cap total 2-hop calls per (arm,target).
            seen_q: Set[tuple[str, str, str]] = set()
            capped: List[tuple[str, List[tuple[str, str]]]] = []
            for ent, rels in second_hop_queries:
                if not rels:
                    continue
                pred, pos = rels[0]
                key = (ent, pred, pos)
                if key in seen_q:
                    continue
                seen_q.add(key)
                capped.append((ent, rels))
                if len(capped) >= 5:
                    break

            for ent, rels in capped:
                _, cap_reached = _do_retrieve(ent, rels)
                if cap_reached:
                    return out_triples, provenance, web_entities_dict

        return out_triples, provenance, web_entities_dict


def restore_original_dataset(dir_triples: str) -> bool:
    """Restore dataset from backup.
    
    Args:
        dir_triples: Directory containing the current (possibly modified) dataset.
        
    Returns:
        True if restoration was successful, False otherwise.
    """
    import shutil
    dataset_path = Path(dir_triples)
    backup_path = dataset_path.parent / f"{dataset_path.name}_original_backup"
    
    if not backup_path.exists():
        logger.error(f"Backup not found at: {backup_path}")
        return False
    
    try:
        logger.info(f"Restoring dataset from backup: {backup_path} -> {dataset_path}")
        # Remove current dataset
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        # Copy backup to original location
        shutil.copytree(backup_path, dataset_path)
        logger.info(f"Dataset restored successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to restore dataset: {e}")
        return False
