"""Arm history management module.

This module records the effect of each arm (a combination of rules) per
iteration and provides aggregated statistics over time.

Design principles:
- Follow Google Style Docstring
- PEP8 compliance
- Explicit type hints
- Use util.get_logger() for consistent logging
"""

from __future__ import annotations

import json
import os
import pickle
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from simple_active_refine.arm import Arm
from simple_active_refine.util import get_logger

logger = get_logger("arm_history")

Triple = Tuple[str, str, str]


@dataclass
class ArmEvaluationRecord:
    """Evaluation record for a single arm in a single iteration.

    Attributes:
        iteration: Iteration index.
        arm_id: Stable arm identifier.
        arm: Arm object.
        target_triples: Target triples evaluated/considered for this arm.
        added_triples: Triples added to the KG as a result of this arm.
        reward: Proxy reward assigned to this arm.
        diagnostics: Optional diagnostic metrics (e.g., witness counts, conflicts).
    """

    iteration: int
    arm_id: str
    arm: Arm
    target_triples: List[Triple]
    added_triples: List[Triple]
    reward: float
    diagnostics: Dict[str, float]

    # Optional richer context for selection/prompting.
    # - evidence_triples: all evidence/body triples acquired for this arm in this iteration
    # - witness_by_target: per-target witness counts used to judge explanatory power
    evidence_triples: List[Triple] = field(default_factory=list)
    witness_by_target: Dict[Triple, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert record to a JSON-serializable dictionary.

        Returns:
            Dict: JSON-serializable dictionary.
        """

        return {
            "iteration": self.iteration,
            "arm_id": self.arm_id,
            "arm": {
                "arm_type": self.arm.arm_type,
                "rule_keys": list(self.arm.rule_keys),
                "metadata": dict(self.arm.metadata),
            },
            "target_triples": [list(t) for t in self.target_triples],
            "added_triples": [list(t) for t in self.added_triples],
            "evidence_triples": [list(t) for t in self.evidence_triples],
            "witness_by_target": {"\t".join(t): int(w) for t, w in self.witness_by_target.items()},
            "reward": self.reward,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class ArmStatistics:
    """Aggregated statistics for an arm.

    Attributes:
        arm_id: Stable arm identifier.
        total_iterations: Number of iterations the arm was evaluated.
        total_triples_added: Total number of triples added across records.
        mean_reward: Mean reward.
        std_reward: Standard deviation of reward.
        recent_performance: Mean reward over the most recent N evaluations.
    """

    arm_id: str
    total_iterations: int
    total_triples_added: int
    mean_reward: float
    std_reward: float
    recent_performance: float

    def __repr__(self) -> str:
        return (
            f"ArmStats(id={self.arm_id}, iters={self.total_iterations}, "
            f"mean_reward={self.mean_reward:.4f}, recent={self.recent_performance:.4f}, "
            f"total_added={self.total_triples_added})"
        )


class ArmHistory:
    """Tracks arm evaluation records and computes arm-level statistics."""

    def __init__(self) -> None:
        self.records: List[ArmEvaluationRecord] = []
        self._records_by_arm: Dict[str, List[ArmEvaluationRecord]] = {}
        self._records_by_iteration: Dict[int, List[ArmEvaluationRecord]] = {}

    def add_record(self, record: ArmEvaluationRecord) -> None:
        """Add a new evaluation record.

        Args:
            record: ArmEvaluationRecord to add.
        """

        self.records.append(record)

        if record.arm_id not in self._records_by_arm:
            self._records_by_arm[record.arm_id] = []
        self._records_by_arm[record.arm_id].append(record)

        if record.iteration not in self._records_by_iteration:
            self._records_by_iteration[record.iteration] = []
        self._records_by_iteration[record.iteration].append(record)

        logger.debug(
            "Added arm record: iter=%d, arm=%s, reward=%.4f",
            record.iteration,
            record.arm_id,
            record.reward,
        )

    def get_records_for_arm(self, arm_id: str) -> List[ArmEvaluationRecord]:
        """Get all records for a specific arm.

        Args:
            arm_id: Stable arm identifier.

        Returns:
            List[ArmEvaluationRecord]: Records for this arm.
        """

        return self._records_by_arm.get(arm_id, [])

    def get_records_for_iteration(self, iteration: int) -> List[ArmEvaluationRecord]:
        """Get all records for a specific iteration.

        Args:
            iteration: Iteration index.

        Returns:
            List[ArmEvaluationRecord]: Records for this iteration.
        """

        return self._records_by_iteration.get(iteration, [])

    def get_arm_statistics(self, arm_id: str, recent_n: int = 3) -> Optional[ArmStatistics]:
        """Compute aggregated statistics for a specific arm.

        Args:
            arm_id: Stable arm identifier.
            recent_n: Number of recent evaluations to use for recent_performance.

        Returns:
            Optional[ArmStatistics]: Statistics or None if no records exist.
        """

        records = self.get_records_for_arm(arm_id)
        if not records:
            return None

        rewards = [rec.reward for rec in records]
        total_added = sum(len(rec.added_triples) for rec in records)

        recent_records = records[-recent_n:]
        recent_rewards = [rec.reward for rec in recent_records]

        return ArmStatistics(
            arm_id=arm_id,
            total_iterations=len(records),
            total_triples_added=total_added,
            mean_reward=statistics.mean(rewards) if rewards else 0.0,
            std_reward=statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            recent_performance=statistics.mean(recent_rewards) if recent_rewards else 0.0,
        )

    def get_all_arm_statistics(self, recent_n: int = 3) -> Dict[str, ArmStatistics]:
        """Compute statistics for all tracked arms.

        Args:
            recent_n: Number of recent evaluations to use for recent_performance.

        Returns:
            Dict[str, ArmStatistics]: Mapping arm_id -> statistics.
        """

        stats: Dict[str, ArmStatistics] = {}
        for arm_id in self._records_by_arm.keys():
            stat = self.get_arm_statistics(arm_id, recent_n=recent_n)
            if stat is not None:
                stats[arm_id] = stat
        return stats

    def save(self, filepath: str) -> None:
        """Save history to a pickle file.

        Args:
            filepath: Destination path.
        """

        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved ArmHistory to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> ArmHistory:
        """Load history from a pickle file.

        Args:
            filepath: Source path.

        Returns:
            ArmHistory: Loaded history.
        """

        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not ArmHistory: {type(obj)}")
        return obj

    def save_json(self, filepath: str) -> None:
        """Save history to a JSON file.

        Args:
            filepath: Destination path.
        """

        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        data = [rec.to_dict() for rec in self.records]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved ArmHistory JSON to %s", filepath)
