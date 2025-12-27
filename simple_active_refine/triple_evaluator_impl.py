"""Concrete triple evaluator implementations."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from simple_active_refine.pipeline import BaseTripleEvaluator, TripleAcquisitionResult, TripleEvaluationContext, TripleEvaluationResult

Triple = Tuple[str, str, str]


class AcceptAllTripleEvaluator(BaseTripleEvaluator):
    """Accept all candidates and compute simple diagnostics (no filtering)."""

    def __init__(self, dump_base_dir: Optional[str] = None) -> None:
        self.dump_base_dir = dump_base_dir

    def evaluate(
        self,
        context: TripleEvaluationContext,
        acquisition: TripleAcquisitionResult,
    ) -> TripleEvaluationResult:
        accepted: List[Triple] = []
        rule_rewards: Dict[str, float] = {}

        for rule_key, triples in acquisition.candidates_by_rule.items():
            rule_rewards[rule_key] = float(len(triples))
            accepted.extend(triples)

        unique = list(dict.fromkeys([tuple(t) for t in accepted]))
        triple_scores = {t: 1.0 for t in unique}

        diagnostics = {
            "n_accepted": len(unique),
            "n_rules_rewarded": len(rule_rewards),
        }
        if self.dump_base_dir:
            iter_dir = os.path.join(self.dump_base_dir, f"iter_{context.iteration}")
            os.makedirs(iter_dir, exist_ok=True)
            dump_path = os.path.join(iter_dir, "triple_evaluator_io.json")
            try:
                payload = {
                    "iteration": context.iteration,
                    "input": {
                        "rules": list(acquisition.candidates_by_rule.keys()),
                        "candidates_by_rule": {k: [list(t) for t in v] for k, v in acquisition.candidates_by_rule.items()},
                    },
                    "output": {
                        "accepted_triples": [list(t) for t in unique],
                        "rejected_triples": [],
                        "rule_rewards": rule_rewards,
                        "triple_scores": {"|".join(t): s for t, s in triple_scores.items()},
                        "diagnostics": diagnostics,
                    },
                }
                with open(dump_path, "w", encoding="utf-8") as fout:
                    json.dump(payload, fout, ensure_ascii=False, indent=2)
            except Exception as err:
                # Best-effort: do not break pipeline on dump failure
                pass
        return TripleEvaluationResult(
            accepted_triples=unique,
            rejected_triples=[],
            rule_rewards=rule_rewards,
            triple_scores=triple_scores,
            diagnostics=diagnostics,
        )
