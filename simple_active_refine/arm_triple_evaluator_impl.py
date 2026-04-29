"""Arm-level evaluation implementations.

This module defines proxy evaluation functions for selected arms based on
witness counts and evidence additions.

Note:
    Target relation triples (r_*) are treated as hypothesis and are store-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from simple_active_refine.arm_triple_acquirer_impl import ArmAcquisitionResult
from simple_active_refine.util import get_logger

logger = get_logger("arm_triple_evaluator")

Triple = Tuple[str, str, str]


@dataclass
class ArmEvaluationResult:
    """Evaluation results for an iteration."""

    accepted_evidence_triples: List[Triple]
    pending_hypothesis_triples: List[Triple]
    reward_by_arm: Dict[str, float]
    diagnostics: Dict


class WitnessConflictArmEvaluator:
    """Proxy evaluator using witness counts and accepted evidence size.

    v1 behavior:
    - Evidence triples are the only triples added to the KG.
    - Target relation (r_*) triples are store-only and written to pending.
    - Conflict handling is reserved for future expansion; conflict_count=0.
    """

    def __init__(
        self,
        witness_weight: float = 1.0,
        evidence_weight: float = 1.0,
        hypothesis_predicates: Sequence[str] | None = None,
    ) -> None:
        self.witness_weight = witness_weight
        self.evidence_weight = evidence_weight
        self.hypothesis_predicates = set(hypothesis_predicates or [])

    def evaluate(
        self,
        acquisition: ArmAcquisitionResult,
        current_kg_triples: Iterable[Triple],
        prev_kge_scores: Optional[Dict[Triple, float]] = None,
    ) -> ArmEvaluationResult:
        """Evaluate acquisition results to compute rewards and accept evidence.

        Reward formula (when ``prev_kge_scores`` and novelty-witness are available):

        .. math::
            R(a) = \\sum_{t} (1 - \\hat{s}_{\\text{prev}}(t))
                   \\cdot \\text{novelty-witness}(a, t)
                   + \\lambda_e \\cdot |\\text{NewEvidence}(a)|

        When ``prev_kge_scores`` is *not* provided, the legacy weighted-witness
        formula is used as a fallback (backward compatible):

        .. math::
            R(a) = \\lambda_w \\cdot \\text{witness\\_sum} + \\lambda_e \\cdot |\\text{NewEvidence}(a)|

        Args:
            acquisition: ArmAcquisitionResult from an acquirer.
            current_kg_triples: Current KG triples.
            prev_kge_scores: Optional mapping from target triple to its KGE score
                (normalised to [0, 1]) from the **previous** KGE training.
                When provided, the new reward formula is activated.

        Returns:
            ArmEvaluationResult
        """

        current_set = set(current_kg_triples)

        accepted_evidence_set = set()
        reward_by_arm: Dict[str, float] = {}

        witness_total_all = 0.0
        witness_total_raw_all = 0.0
        novelty_witness_total_all = 0.0
        accepted_total_all = 0

        # Determine whether to use KGE-score-weighted novelty-witness reward.
        novelty_map = getattr(acquisition, "novelty_witness_by_arm_and_target", None) or {}
        use_novelty_reward = (
            prev_kge_scores is not None
            and bool(novelty_map)
        )

        for arm_id, evidence in acquisition.evidence_by_arm.items():
            accepted_for_arm = [t for t in evidence if t not in current_set]
            accepted_evidence_set.update(accepted_for_arm)

            witness_raw_sum = float(sum(acquisition.witness_by_arm_and_target.get(arm_id, {}).values()))
            witness_total_raw_all += witness_raw_sum

            weighted_map = getattr(acquisition, "witness_score_by_arm_and_target", None) or {}
            if arm_id in weighted_map:
                witness_sum = float(sum(weighted_map.get(arm_id, {}).values()))
            else:
                witness_sum = witness_raw_sum
            witness_total_all += float(witness_sum)

            accepted_count = len(accepted_for_arm)
            accepted_total_all += accepted_count

            if use_novelty_reward:
                # New formula: Σ_t (1 - s_prev(t)) * novelty_witness(a, t)  +  λ_e * |NewEvidence|
                nw_map = novelty_map.get(arm_id, {})
                novelty_witness_sum = 0.0
                for t, nw in nw_map.items():
                    gap = 1.0 - float((prev_kge_scores or {}).get(t, 0.0))
                    novelty_witness_sum += float(gap) * float(nw)
                novelty_witness_total_all += novelty_witness_sum
                reward = (
                    self.witness_weight * novelty_witness_sum
                    + self.evidence_weight * float(accepted_count)
                )
            else:
                # Legacy formula: λ_w * witness_sum + λ_e * accepted_count
                reward = self.witness_weight * float(witness_sum) + self.evidence_weight * float(accepted_count)
            reward_by_arm[arm_id] = reward

        pending_hypothesis_set = set()
        for arm_id, tmap in acquisition.witness_by_arm_and_target.items():
            for t, w in tmap.items():
                if w <= 0:
                    continue
                if t in current_set:
                    continue
                if not self.hypothesis_predicates or t[1] in self.hypothesis_predicates:
                    pending_hypothesis_set.add(t)

        diagnostics = {
            "witness_total": float(witness_total_all),
            "witness_total_raw": float(witness_total_raw_all),
            "novelty_witness_total": float(novelty_witness_total_all),
            "use_novelty_reward": bool(use_novelty_reward),
            "accepted_evidence_total": float(len(accepted_evidence_set)),
            "accepted_evidence_total_attributed": float(accepted_total_all),
            "conflict_count": 0.0,
        }

        logger.info(
            "Evaluation: accepted_evidence=%d pending_hypothesis=%d witness_total=%.1f",
            len(accepted_evidence_set),
            len(pending_hypothesis_set),
            float(witness_total_all),
        )

        return ArmEvaluationResult(
            accepted_evidence_triples=sorted(accepted_evidence_set),
            pending_hypothesis_triples=sorted(pending_hypothesis_set),
            reward_by_arm=reward_by_arm,
            diagnostics=diagnostics,
        )
