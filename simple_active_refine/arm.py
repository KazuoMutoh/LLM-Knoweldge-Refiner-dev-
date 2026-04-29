from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


ArmType = Literal["set", "sequence"]


@dataclass
class Arm:
    """Combination of rules treated as a single arm.

    Attributes:
        arm_type: "set" for unordered simultaneous application, "sequence" for ordered.
        rule_keys: Rule identifiers (stringified rules or stable rule ids).
        metadata: Optional auxiliary info (e.g., construction method, scores).
    """

    arm_type: ArmType
    rule_keys: List[str]
    metadata: Dict = field(default_factory=dict)

    def key(self) -> str:
        """Return a deterministic key for de-duplication."""
        if self.arm_type == "set":
            return f"set:{','.join(sorted(self.rule_keys))}"
        return f"seq:{','.join(self.rule_keys)}"


@dataclass
class ArmWithId:
    """Arm with stable identifier for selection/history purposes."""

    arm_id: str
    arm: Arm

    @classmethod
    def create(cls, arm: Arm, arm_id: Optional[str] = None, hash_len: int = 12) -> ArmWithId:
        """Create an ArmWithId with deterministic ID.

        Args:
            arm: Arm object.
            arm_id: Optional explicit ID. If None, a deterministic ID is generated.
            hash_len: Length of the hex digest suffix to use.

        Returns:
            ArmWithId: Instance with stable arm_id.
        """
        if arm_id is None:
            key = arm.key()
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()  # nosec - non-cryptographic ID
            arm_id = f"arm_{digest[:hash_len]}"
        return cls(arm_id=arm_id, arm=arm)

