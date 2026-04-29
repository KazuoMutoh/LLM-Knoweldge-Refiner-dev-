"""Relation prior (KGE-friendly) utilities.

This module provides helpers to load and normalize relation-level priors used to
weight witness counts in proxy evaluation.

The expected input is a JSON file mapping predicate -> score, where score is
either:
- a number (already the final prior), or
- an object containing a numeric field like "X" (recommended), or
- an object containing "prior".

All priors are clamped to [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _extract_prior_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("X", "prior", "x"):
            v = value.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def load_relation_priors(path: str | Path) -> Dict[str, float]:
    """Load relation priors from JSON.

    Args:
        path: Path to JSON.

    Returns:
        Dict[str, float]: predicate -> prior in [0, 1].

    Raises:
        ValueError: If the JSON does not contain a dict at top level.
    """

    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"relation priors JSON must be an object, got {type(obj)}")

    # Accept payload format produced by compute_relation_priors.py:
    # {"meta": {...}, "priors": {predicate: {"X": ...}}}
    if "priors" in obj and isinstance(obj.get("priors"), dict):
        obj = obj["priors"]

    out: Dict[str, float] = {}
    for pred, raw in obj.items():
        prior = _extract_prior_value(raw)
        if prior is None:
            continue
        if not isinstance(pred, str) or not pred:
            continue
        out[pred] = clamp01(prior)

    return out
