"""Arm-level triple acquisition implementations.

This module provides local acquisition of evidence/body triples for selected arms
from a candidate triple pool (e.g., train_removed.txt).

Design principles:
- Google Style Docstring
- PEP8 compliance
- Explicit type hints
- Use util.get_logger() for consistent logging
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from simple_active_refine.amie import AmieRule
from simple_active_refine.arm import ArmWithId
from simple_active_refine.triples_editor import (
    TripleIndex,
    count_novelty_witnesses_for_head,
    count_witnesses_for_head,
    find_body_triples_for_head,
)
from simple_active_refine.util import get_logger

logger = get_logger("arm_triple_acquirer")

Triple = Tuple[str, str, str]


@dataclass
class ArmAcquisitionResult:
    """Result of acquiring evidence for a set of selected arms."""

    evidence_by_arm: Dict[str, List[Triple]]
    targets_by_arm: Dict[str, List[Triple]]
    witness_by_arm_and_target: Dict[str, Dict[Triple, int]]
    # Optional weighted witness score (e.g., KGE-friendly relation prior × witness count).
    witness_score_by_arm_and_target: Dict[str, Dict[Triple, float]]
    provenance_by_triple: Dict[Triple, Dict]
    # Novelty-witness: count of groundings that use at least one candidate triple.
    # Empty when candidate_set is not provided to acquire().
    novelty_witness_by_arm_and_target: Dict[str, Dict[Triple, int]]


class LocalArmTripleAcquirer:
    """Acquire evidence/body triples for selected arms from local candidates."""

    def __init__(
        self,
        n_targets_per_arm: int = 50,
        max_witness_per_head: Optional[int] = None,
        relation_priors: Optional[Dict[str, float]] = None,
        default_relation_prior: float = 1.0,
        random_seed: int = 0,
        provenance_source: str = "local",
    ) -> None:
        self.n_targets_per_arm = n_targets_per_arm
        self.max_witness_per_head = max_witness_per_head
        self.relation_priors = dict(relation_priors or {})
        self.default_relation_prior = float(default_relation_prior)
        self.random_seed = random_seed
        self.provenance_source = provenance_source

    def _rule_weight(self, rule: AmieRule) -> float:
        """Compute witness weight for a rule based on body predicates.

        Args:
            rule: AMIE rule.

        Returns:
            float: Weight (>=0). Defaults to 1.0 when priors are not provided.
        """

        w = 1.0
        if not self.relation_priors:
            return w
        for tp in rule.body:
            prior = self.relation_priors.get(tp.p)
            if prior is None:
                prior = self.default_relation_prior
            w *= float(prior)
        return float(w)

    def acquire(
        self,
        selected_arms: Sequence[ArmWithId],
        target_triples: Sequence[Triple],
        candidates: Iterable[Triple] | TripleIndex,
        rule_by_key: Dict[str, AmieRule],
        iteration: int,
        provided_targets_by_arm: Optional[Dict[str, List[Triple]]] = None,
        candidate_set: Optional[Set[Triple]] = None,
    ) -> ArmAcquisitionResult:
        """Acquire evidence triples and witness counts.

        Args:
            selected_arms: Arms selected for this iteration.
            target_triples: Universe of target triples (mainly (x, r*, y)).
            candidates: Candidate triple pool as iterable or pre-built TripleIndex.
            rule_by_key: Mapping from rule key (string) to AmieRule.
            iteration: Iteration index (used for deterministic target sampling).
            provided_targets_by_arm: Optional fixed target mapping per arm.
            candidate_set: Set of triples that are "new" candidates (not in current KG).
                When provided, novelty-witness counts are computed per target.

        Returns:
            ArmAcquisitionResult: Acquired evidence and witness stats.
        """

        idx = candidates if isinstance(candidates, TripleIndex) else TripleIndex(candidates)

        rng = random.Random(self.random_seed + int(iteration))
        targets_all = list(target_triples)

        evidence_by_arm: Dict[str, List[Triple]] = {}
        targets_by_arm: Dict[str, List[Triple]] = {}
        witness_by_arm_and_target: Dict[str, Dict[Triple, int]] = {}
        witness_score_by_arm_and_target: Dict[str, Dict[Triple, float]] = {}
        novelty_witness_by_arm_and_target: Dict[str, Dict[Triple, int]] = {}
        provenance_by_triple: Dict[Triple, Dict] = {}

        for awi in selected_arms:
            arm_id = awi.arm_id
            arm = awi.arm

            if provided_targets_by_arm is not None and arm_id in provided_targets_by_arm:
                selected_targets = list(provided_targets_by_arm.get(arm_id) or [])
            else:
                n = min(self.n_targets_per_arm, len(targets_all))
                selected_targets = rng.sample(targets_all, n) if n > 0 else []
            targets_by_arm[arm_id] = selected_targets

            evidence_set = set()
            witness_map: Dict[Triple, int] = {}
            witness_score_map: Dict[Triple, float] = {}
            novelty_witness_map: Dict[Triple, int] = {}

            for t in selected_targets:
                witness_total = 0
                witness_score_total = 0.0
                novelty_total = 0
                for rule_key in arm.rule_keys:
                    rule = rule_by_key.get(rule_key)
                    if rule is None:
                        raise KeyError(f"Rule key not found in rule_by_key: {rule_key}")

                    w = count_witnesses_for_head(
                        t,
                        rule,
                        idx,
                        max_witness=self.max_witness_per_head,
                    )

                    witness_total += int(w)
                    witness_score_total += float(self._rule_weight(rule)) * float(w)

                    # Novelty-witness: groundings that use at least one candidate triple.
                    if candidate_set is not None:
                        nw = count_novelty_witnesses_for_head(
                            t,
                            rule,
                            idx,
                            candidate_set=candidate_set,
                            max_witness=self.max_witness_per_head,
                        )
                        novelty_total += int(nw)

                    body_triples = find_body_triples_for_head(t, [rule], idx)
                    evidence_set.update(body_triples)

                witness_map[t] = witness_total
                witness_score_map[t] = float(witness_score_total)
                novelty_witness_map[t] = novelty_total

            evidence = sorted(evidence_set)
            evidence_by_arm[arm_id] = evidence
            witness_by_arm_and_target[arm_id] = witness_map
            witness_score_by_arm_and_target[arm_id] = witness_score_map
            novelty_witness_by_arm_and_target[arm_id] = novelty_witness_map

            for tr in evidence:
                provenance_by_triple.setdefault(
                    tr,
                    {
                        "source": self.provenance_source,
                        "iteration": int(iteration),
                    },
                )

            logger.info(
                "Acquired: arm=%s targets=%d evidence=%d",
                arm_id,
                len(selected_targets),
                len(evidence),
            )

        return ArmAcquisitionResult(
            evidence_by_arm=evidence_by_arm,
            targets_by_arm=targets_by_arm,
            witness_by_arm_and_target=witness_by_arm_and_target,
            witness_score_by_arm_and_target=witness_score_by_arm_and_target,
            provenance_by_triple=provenance_by_triple,
            novelty_witness_by_arm_and_target=novelty_witness_by_arm_and_target,
        )
