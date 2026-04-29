from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from simple_active_refine.amie import AmieRule
from simple_active_refine.arm import Arm, ArmWithId
from simple_active_refine.triples_editor import TripleIndex, count_witnesses_for_head, supports_head
from simple_active_refine.util import get_logger

logger = get_logger("arm_builder")

Triple = Tuple[str, str, str]


@dataclass
class ArmBuilderConfig:
    """Configuration for building initial arms.

    Attributes:
        arm_type: "set" or "sequence" (only "set" used for now).
        k_pairs: Number of top co-occurring pairs to keep.
        max_witness_per_head: Optional cap when counting witnesses.
        pair_support_source: Which triple set to use when computing pair-arm support/Jaccard.
            - "candidate": use candidate_triples (default, backward compatible; typically train_removed)
            - "train": use pair_support_triples (typically train.txt)
    """

    arm_type: str = "set"
    k_pairs: int = 20
    max_witness_per_head: Optional[int] = None
    pair_support_source: str = "candidate"


def _rule_key(rule: AmieRule) -> str:
    return str(rule)


def build_initial_arms(
    rule_pool: Sequence[AmieRule],
    target_triples: Sequence[Triple],
    candidate_triples: Sequence[Triple],
    config: Optional[ArmBuilderConfig] = None,
    pair_support_triples: Optional[Sequence[Triple]] = None,
) -> List[Arm]:
    """Build initial arms using head-based support on target triples.

    - Singleton arms for every rule
    - Pair arms for top co-occurring rules based on shared supported targets

    Args:
        rule_pool: List of rules (body=2 expected).
        target_triples: Target triples (head predicate should be r*).
        candidate_triples: Triples used to satisfy rule bodies (e.g., train + removed).
        config: ArmBuilderConfig

    Returns:
        List of Arm objects (singleton + top pairs).
    """

    cfg = config or ArmBuilderConfig()
    if cfg.arm_type != "set":
        raise ValueError("Only set-type arms are supported in initial builder")

    if cfg.pair_support_source not in {"candidate", "train"}:
        raise ValueError(
            f"Unsupported pair_support_source={cfg.pair_support_source!r} (expected 'candidate' or 'train')"
        )

    if cfg.pair_support_source == "train":
        if pair_support_triples is None:
            raise ValueError(
                "pair_support_source='train' requires pair_support_triples (e.g., train.txt triples)"
            )
        support_triples = pair_support_triples
    else:
        support_triples = candidate_triples

    logger.info(
        "[arm_builder] building arms: rules=%d, targets=%d, candidates=%d, pair_support_source=%s, pair_support_triples=%d",
        len(rule_pool),
        len(target_triples),
        len(candidate_triples),
        cfg.pair_support_source,
        len(support_triples),
    )

    idx = TripleIndex(support_triples)

    # 1) singleton arms
    singletons: List[Arm] = [Arm(arm_type="set", rule_keys=[_rule_key(r)], metadata={"kind": "singleton"}) for r in rule_pool]

    # 2) support sets per rule (which target triples are supported)
    support_sets: Dict[str, List[Triple]] = {}
    for rule in rule_pool:
        key = _rule_key(rule)
        supported: List[Triple] = []
        for t in target_triples:
            if cfg.max_witness_per_head:
                w = count_witnesses_for_head(t, rule, idx, max_witness=cfg.max_witness_per_head)
                if w > 0:
                    supported.append(t)
            else:
                if supports_head(t, rule, idx):
                    supported.append(t)
        support_sets[key] = supported
        logger.debug("[arm_builder] rule support: %s -> %d", key, len(supported))

    # 3) pair arms using Jaccard on supported targets
    pair_scores: List[Tuple[float, Tuple[str, str]]] = []
    keys = list(support_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            si, sj = set(support_sets[ki]), set(support_sets[kj])
            if not si or not sj:
                continue
            inter = si & sj
            union = si | sj
            if not union:
                continue
            score = len(inter) / len(union)
            if score > 0:
                pair_scores.append((score, (ki, kj)))

    pair_scores.sort(key=lambda x: x[0], reverse=True)
    k_pairs = min(cfg.k_pairs, len(pair_scores))
    pairs = [pair_scores[i][1] for i in range(k_pairs)]

    pair_arms = [Arm(arm_type="set", rule_keys=list(p), metadata={"kind": "pair", "cooc": pair_scores[idx][0]}) for idx, p in enumerate(pairs)]

    arms = singletons + pair_arms
    logger.info(
        "[arm_builder] built arms: singleton=%d, pairs=%d (kept %d of %d pairs)",
        len(singletons), len(pair_arms), k_pairs, len(pair_scores),
    )
    return arms


def save_arms_json(arms: Iterable[Arm], path: str) -> None:
    data = []
    for a in arms:
        data.append({"arm_type": a.arm_type, "rule_keys": a.rule_keys, "metadata": a.metadata})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_arms_json(path: str | Path) -> List[Arm]:
    """Load arms from the JSON output produced by build_initial_arms.py.

    Args:
        path: Path to initial_arms.json.

    Returns:
        List of Arm.
    """

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Arms JSON must be a list, got {type(raw)}")

    arms: List[Arm] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Arms JSON item must be dict, got {type(item)} at index {i}")
        arm_type = item.get("arm_type")
        rule_keys = item.get("rule_keys")
        metadata = item.get("metadata") or {}
        if arm_type is None or rule_keys is None:
            raise ValueError(f"Missing arm_type/rule_keys at index {i}")
        if not isinstance(rule_keys, list):
            raise ValueError(f"rule_keys must be a list at index {i}, got {type(rule_keys)}")
        arms.append(Arm(arm_type=arm_type, rule_keys=list(rule_keys), metadata=dict(metadata)))
    return arms


def load_arms_pickle(path: str | Path) -> List[Arm]:
    """Load arms from the pickle output produced by build_initial_arms.py.

    Args:
        path: Path to initial_arms.pkl.

    Returns:
        List of Arm.
    """

    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"Expected list in arms pickle, got {type(obj)}")
    for i, a in enumerate(obj):
        if not isinstance(a, Arm):
            raise TypeError(f"Expected Arm at index {i}, got {type(a)}")
    return obj


def load_arm_pool_with_ids(path: str | Path) -> List[ArmWithId]:
    """Load initial arms (json/pkl) and attach deterministic arm IDs.

    Args:
        path: Path to initial_arms.json or initial_arms.pkl.

    Returns:
        List of ArmWithId.
    """

    p = Path(path)
    if p.suffix.lower() == ".json":
        arms = load_arms_json(p)
    elif p.suffix.lower() == ".pkl":
        arms = load_arms_pickle(p)
    else:
        raise ValueError(f"Unsupported arms file extension: {p.suffix} (expected .json or .pkl)")

    return [ArmWithId.create(a) for a in arms]

