"""Compute relation priors (KGE-friendly scores).

This module implements the metrics described in:
- docs/external/KGEフレンドさを考慮したwitness評価の改善.md

It computes:
- X_r(2): hubness penalty proxy (higher is better)
- X_r(3): pseudo type/role coherence via predicate-distribution cosine similarity
- X_r(4): pattern concentration proxy via relation-specific in/out fan-out
- X_r(7): geometric consistency (TransE-style) via variance of (e_t - e_h)

Then aggregates them into a final prior X_r via a weighted sum.

Notes
-----
- X_r(7) requires a trained KGE model directory containing trained_model.pkl and
  training_triples/ saved by PyKEEN (PipelineResult.save_to_directory()).
- Other metrics only require train triples.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.io_utils import read_triples
from simple_active_refine.util import get_logger

logger = get_logger("relation_priors_compute")

Triple = Tuple[str, str, str]


@dataclass(frozen=True)
class RelationPriorConfig:
    """Configuration for computing relation priors."""

    # Sampling limits to keep runtime reasonable
    max_samples_x3_per_relation: int = 2000
    max_samples_x7_per_relation: int = 5000
    random_seed: int = 0

    # X7 reliability
    min_count_x7: int = 50

    # Aggregation weights (weighted average over available components)
    # Default: emphasize geometric consistency only.
    weight_x2: float = 0.0
    weight_x3: float = 0.0
    weight_x4: float = 0.0
    weight_x7: float = 1.0


def _group_by_relation(triples: Sequence[Triple]) -> Dict[str, List[Triple]]:
    by_r: Dict[str, List[Triple]] = {}
    for h, r, t in triples:
        by_r.setdefault(r, []).append((h, r, t))
    return by_r


def _minmax01(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    if math.isclose(vmin, vmax):
        return {k: 1.0 for k in values.keys()}
    out: Dict[str, float] = {}
    for k, v in values.items():
        x = (v - vmin) / (vmax - vmin)
        out[k] = float(max(0.0, min(1.0, x)))
    return out


def _sample_triples(
    triples: Sequence[Triple],
    max_samples: int,
    rng: random.Random,
) -> Sequence[Triple]:
    if max_samples <= 0:
        return []
    if len(triples) <= max_samples:
        return triples
    idxs = rng.sample(range(len(triples)), max_samples)
    return [triples[i] for i in idxs]


def compute_x2_hubness(triples: Sequence[Triple]) -> Dict[str, float]:
    """Compute X_r(2) hubness proxy.

    Uses global degree deg(v) = in+out over all train triples, and for each
    relation r, computes:
        raw(r) = 1 / E_{(h,r,t)}[log(2 + deg(t))]
    Then min-max normalizes raw(r) to [0,1].

    Returns:
        Dict[predicate, score in [0,1]]
    """

    deg: Dict[str, int] = {}
    for h, _, t in triples:
        deg[h] = deg.get(h, 0) + 1
        deg[t] = deg.get(t, 0) + 1

    by_r = _group_by_relation(triples)
    raw: Dict[str, float] = {}
    for r, trs in by_r.items():
        if not trs:
            continue
        vals = []
        for _, _, tail in trs:
            vals.append(math.log(2.0 + float(deg.get(tail, 0))))
        mean = float(sum(vals) / float(len(vals))) if vals else 0.0
        if mean <= 0.0:
            continue
        raw[r] = 1.0 / mean

    return _minmax01(raw)


def compute_x4_concentration(triples: Sequence[Triple]) -> Dict[str, float]:
    """Compute X_r(4) pattern concentration proxy.

    For each relation r:
      out_count(h) = |{t : (h,r,t)}|
      in_count(t)  = |{h : (h,r,t)}|

    Score is:
      X4(r) = 0.5*E_h[1/(1+out_count(h))] + 0.5*E_t[1/(1+in_count(t))]

    Returns:
        Dict[predicate, score in [0,1]]
    """

    by_r = _group_by_relation(triples)
    out: Dict[str, float] = {}

    for r, trs in by_r.items():
        out_deg: Dict[str, int] = {}
        in_deg: Dict[str, int] = {}
        for h, _, t in trs:
            out_deg[h] = out_deg.get(h, 0) + 1
            in_deg[t] = in_deg.get(t, 0) + 1

        if out_deg:
            out_mean = float(sum(1.0 / (1.0 + float(c)) for c in out_deg.values()) / float(len(out_deg)))
        else:
            out_mean = 0.0

        if in_deg:
            in_mean = float(sum(1.0 / (1.0 + float(c)) for c in in_deg.values()) / float(len(in_deg)))
        else:
            in_mean = 0.0

        out[r] = float(0.5 * out_mean + 0.5 * in_mean)

    # Already in [0,1]
    return {k: float(max(0.0, min(1.0, v))) for k, v in out.items()}


def _build_role_signatures(triples: Sequence[Triple]) -> tuple[Dict[str, Dict[str, int]], Dict[str, float]]:
    """Build sparse role signatures S(v) and their L2 norms.

    S(v) is a sparse vector over keys "out:<p>" and "in:<p>".

    Returns:
        (sig, norms)
        sig[v] -> dict[key -> count]
        norms[v] -> L2 norm
    """

    sig: Dict[str, Dict[str, int]] = {}

    def inc(entity: str, key: str) -> None:
        d = sig.setdefault(entity, {})
        d[key] = d.get(key, 0) + 1

    for h, p, t in triples:
        inc(h, f"out:{p}")
        inc(t, f"in:{p}")

    norms: Dict[str, float] = {}
    for ent, d in sig.items():
        norms[ent] = float(math.sqrt(sum(float(c) * float(c) for c in d.values())))

    return sig, norms


def _cosine_sparse(a: Dict[str, int], b: Dict[str, int], norm_a: float, norm_b: float) -> float:
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, v in a.items():
        vb = b.get(k)
        if vb is None:
            continue
        dot += float(v) * float(vb)
    return float(dot / (norm_a * norm_b))


def compute_x3_role_coherence(
    triples: Sequence[Triple],
    *,
    max_samples_per_relation: int = 2000,
    random_seed: int = 0,
) -> Dict[str, float]:
    """Compute X_r(3) pseudo type/role coherence.

    For each relation r, compute mean cosine similarity between S(h) and S(t)
    over sampled triples.

    Returns:
        Dict[predicate, score in [0,1]]
    """

    by_r = _group_by_relation(triples)
    sig, norms = _build_role_signatures(triples)

    rng = random.Random(int(random_seed))
    out: Dict[str, float] = {}
    for r, trs in by_r.items():
        sampled = _sample_triples(trs, max_samples=max_samples_per_relation, rng=rng)
        if not sampled:
            continue
        vals: List[float] = []
        for h, _, t in sampled:
            vals.append(_cosine_sparse(sig.get(h, {}), sig.get(t, {}), norms.get(h, 0.0), norms.get(t, 0.0)))
        out[r] = float(sum(vals) / float(len(vals))) if vals else 0.0

    # Cosine is in [0,1] here.
    return {k: float(max(0.0, min(1.0, v))) for k, v in out.items()}


def _extract_entity_embedding_matrix(kge: KnowledgeGraphEmbedding) -> np.ndarray:
    model = kge.model
    rep = model.entity_representations[0]
    # In our environment (PyKEEN), this is usually Embedding with _embeddings.weight
    if hasattr(rep, "_embeddings") and hasattr(rep._embeddings, "weight"):
        mat = rep._embeddings.weight.detach().cpu().numpy()
        return mat

    # Fallback: call representation
    import torch

    idx = torch.arange(model.num_entities)
    mat = rep(indices=idx).detach().cpu().numpy()
    return mat


def compute_x7_geometric_consistency(
    *,
    triples: Sequence[Triple],
    entity_to_id: Dict[str, int],
    entity_embeddings: np.ndarray,
    min_count: int = 50,
    max_samples_per_relation: int = 5000,
    random_seed: int = 0,
) -> Dict[str, float]:
    """Compute X_r(7) geometric consistency (TransE-style).

    For each relation r, compute deltas d = e_t - e_h for triples (h,r,t).
    Let mu be the mean delta; compute mean squared deviation:
        var = E[||d - mu||^2]
    Score is:
        X7 = exp(-var)

    Relations with too few usable triples (< min_count) are omitted.

    Returns:
        Dict[predicate, score in (0,1]]
    """

    by_r = _group_by_relation(triples)
    rng = random.Random(int(random_seed))

    out: Dict[str, float] = {}
    for r, trs in by_r.items():
        # Keep only triples that exist in embedding mapping
        filtered: List[Triple] = []
        for h, _, t in trs:
            if h in entity_to_id and t in entity_to_id:
                filtered.append((h, r, t))
        if len(filtered) < int(min_count):
            continue

        sampled = _sample_triples(filtered, max_samples=max_samples_per_relation, rng=rng)
        if not sampled:
            continue

        h_ids = np.fromiter((entity_to_id[h] for h, _, _ in sampled), dtype=np.int64)
        t_ids = np.fromiter((entity_to_id[t] for _, _, t in sampled), dtype=np.int64)

        deltas = entity_embeddings[t_ids] - entity_embeddings[h_ids]
        mu = deltas.mean(axis=0, keepdims=True)
        centered = deltas - mu
        var = float(np.mean(np.sum(centered * centered, axis=1)))

        # exp(-var) is in (0,1], but may underflow if var is huge.
        score = float(math.exp(-var)) if var < 700.0 else 0.0
        out[r] = float(max(0.0, min(1.0, score)))

    return out


def aggregate_relation_priors(
    *,
    relations: Sequence[str],
    x2: Dict[str, float],
    x3: Dict[str, float],
    x4: Dict[str, float],
    x7: Dict[str, float],
    cfg: RelationPriorConfig,
) -> Dict[str, Dict[str, float]]:
    """Aggregate X2/X3/X4/X7 into final X via weighted sum."""

    out: Dict[str, Dict[str, float]] = {}

    for r in relations:
        parts: List[Tuple[str, float, float]] = []
        if r in x2:
            parts.append(("X2", float(cfg.weight_x2), float(x2[r])))
        if r in x3:
            parts.append(("X3", float(cfg.weight_x3), float(x3[r])))
        if r in x4:
            parts.append(("X4", float(cfg.weight_x4), float(x4[r])))
        if r in x7:
            parts.append(("X7", float(cfg.weight_x7), float(x7[r])))

        if not parts:
            continue

        w_sum = float(sum(w for _, w, _ in parts))
        if w_sum > 0.0:
            x = float(sum(w * v for _, w, v in parts) / w_sum)
        else:
            # If all configured weights are zero (or all weighted components are
            # missing), still provide a usable X by falling back to an
            # unweighted mean over available components.
            x = float(sum(v for _, _, v in parts) / float(len(parts)))

        x = float(max(0.0, min(1.0, x)))

        row = {"X": x}
        for name, _, v in parts:
            row[name] = float(max(0.0, min(1.0, v)))
        out[r] = row

    return out


def compute_relation_priors(
    *,
    train_triples: Sequence[Triple],
    kge_before: Optional[KnowledgeGraphEmbedding],
    cfg: RelationPriorConfig,
) -> Dict[str, Dict[str, float]]:
    """Compute relation priors.

    Args:
        train_triples: Train triples.
        kge_before: Optional loaded KGE for X7 computation.
        cfg: Config.

    Returns:
        Mapping predicate -> dict with X and components.
    """

    by_r = _group_by_relation(train_triples)
    relations = sorted(by_r.keys())

    logger.info("Computing X2 (hubness) for %d relations", len(relations))
    x2 = compute_x2_hubness(train_triples)

    logger.info("Computing X3 (role coherence) for %d relations", len(relations))
    x3 = compute_x3_role_coherence(
        train_triples,
        max_samples_per_relation=int(cfg.max_samples_x3_per_relation),
        random_seed=int(cfg.random_seed),
    )

    logger.info("Computing X4 (concentration) for %d relations", len(relations))
    x4 = compute_x4_concentration(train_triples)

    x7: Dict[str, float] = {}
    if kge_before is not None:
        logger.info("Computing X7 (geometric consistency) from KGE before model")
        entity_to_id = dict(kge_before.triples.entity_to_id)
        emb = _extract_entity_embedding_matrix(kge_before)
        x7 = compute_x7_geometric_consistency(
            triples=train_triples,
            entity_to_id=entity_to_id,
            entity_embeddings=emb,
            min_count=int(cfg.min_count_x7),
            max_samples_per_relation=int(cfg.max_samples_x7_per_relation),
            random_seed=int(cfg.random_seed),
        )
    else:
        logger.warning("KGE before model not provided; skipping X7")

    priors = aggregate_relation_priors(
        relations=relations,
        x2=x2,
        x3=x3,
        x4=x4,
        x7=x7,
        cfg=cfg,
    )

    # Add stats
    for r, trs in by_r.items():
        if r in priors:
            priors[r]["n_triples"] = float(len(trs))
            priors[r]["has_x7"] = 1.0 if r in x7 else 0.0

    return priors


def compute_and_save_relation_priors(
    *,
    dataset_dir: str | Path,
    model_before_dir: Optional[str | Path],
    output_path: str | Path,
    cfg: RelationPriorConfig,
) -> Path:
    """Compute priors and write relation_priors.json.

    Args:
        dataset_dir: Directory containing train.txt.
        model_before_dir: Trained KGE directory (optional for X7).
        output_path: Output JSON path.
        cfg: Config.

    Returns:
        Path to written JSON.
    """

    dataset_dir = Path(dataset_dir)
    train_path = dataset_dir / "train.txt"
    train_triples = read_triples(train_path)

    kge_before: Optional[KnowledgeGraphEmbedding] = None
    if model_before_dir is not None:
        kge_before = KnowledgeGraphEmbedding(str(model_before_dir))

    priors = compute_relation_priors(train_triples=train_triples, kge_before=kge_before, cfg=cfg)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "dataset_dir": str(dataset_dir),
            "model_before_dir": str(model_before_dir) if model_before_dir is not None else None,
            "config": {
                "max_samples_x3_per_relation": int(cfg.max_samples_x3_per_relation),
                "max_samples_x7_per_relation": int(cfg.max_samples_x7_per_relation),
                "random_seed": int(cfg.random_seed),
                "min_count_x7": int(cfg.min_count_x7),
                "weight_x2": float(cfg.weight_x2),
                "weight_x3": float(cfg.weight_x3),
                "weight_x4": float(cfg.weight_x4),
                "weight_x7": float(cfg.weight_x7),
            },
        },
        "priors": priors,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Wrote relation priors: %s (relations=%d)", out_path, len(priors))
    return out_path


def load_relation_priors_payload(path: str | Path) -> Dict[str, float]:
    """Load a payload written by compute_and_save_relation_priors() as predicate->X.

    This helper is useful if you want to pass the computed JSON directly into
    the existing loader which expects predicate -> number or predicate -> {X:..}.
    """

    with open(Path(path), "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "priors" in obj and isinstance(obj["priors"], dict):
        return {k: float(v.get("X")) for k, v in obj["priors"].items() if isinstance(v, dict) and "X" in v}
    raise ValueError("Invalid relation priors payload format")
