from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from simple_active_refine.util import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SeedHierarchyResult:
    tau_opt: float
    labels: np.ndarray
    clusters: Dict[int, List[str]]
    cluster_centers: np.ndarray
    cluster_labels: List[int]


@dataclass(frozen=True)
class SeedHierarchyArtifacts:
    entity_ids: Tuple[str, ...]
    entity_cluster_labels: np.ndarray
    cluster_labels: List[int]
    cluster_centers: np.ndarray
    neighbor_clusters: Dict[int, List[int]]


def _cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x_norm = x / norm
    sim = x_norm @ x_norm.T
    return 1.0 - sim


def _silhouette_score(labels: np.ndarray, distance_matrix: np.ndarray) -> float:
    n = labels.shape[0]
    unique_labels = np.unique(labels)
    if unique_labels.size == 1:
        return -1.0

    label_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    scores = []

    for i in range(n):
        label = labels[i]
        same = label_indices[label]
        if same.size <= 1:
            scores.append(0.0)
            continue
        a = np.mean(distance_matrix[i, same[same != i]])
        b = np.inf
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_idx = label_indices[other_label]
            b = min(b, float(np.mean(distance_matrix[i, other_idx])))
        scores.append((b - a) / max(a, b))

    return float(np.mean(scores))


def _select_sample_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_samples, replace=False)


def build_seed_hierarchy(
    *,
    embeddings: np.ndarray,
    entity_ids: Iterable[str],
    tau_min: float = 0.15,
    tau_max: float = 0.85,
    tau_steps: int = 15,
    max_silhouette_samples: int = 2000,
    random_seed: int = 0,
) -> SeedHierarchyResult:
    """Build seed hierarchy using agglomerative clustering with cosine distance.

    Args:
        embeddings: Array of shape (n_entities, dim).
        entity_ids: Entity identifiers in the same order as embeddings rows.
        tau_min: Minimum distance threshold for clustering.
        tau_max: Maximum distance threshold for clustering.
        tau_steps: Number of thresholds to scan.
        max_silhouette_samples: Max samples used for silhouette score.
        random_seed: RNG seed for sampling.

    Returns:
        SeedHierarchyResult with tau_opt, labels, clusters, and cluster centers.
    """

    entity_ids = list(entity_ids)
    if embeddings.shape[0] != len(entity_ids):
        raise ValueError("embeddings and entity_ids must have the same length")

    logger.info("Building linkage (average/cosine) for %d entities", len(entity_ids))
    link = linkage(embeddings, method="average", metric="cosine")

    thresholds = np.linspace(tau_min, tau_max, num=tau_steps)
    best_tau = thresholds[0]
    best_score = -np.inf

    sample_idx = _select_sample_indices(len(entity_ids), max_silhouette_samples, random_seed)
    sample_embeddings = embeddings[sample_idx]
    dist_mat = _cosine_distance_matrix(sample_embeddings)

    for tau in thresholds:
        labels = fcluster(link, t=tau, criterion="distance")
        sample_labels = labels[sample_idx]
        score = _silhouette_score(sample_labels, dist_mat)
        logger.info("tau=%.3f silhouette=%.4f clusters=%d", tau, score, len(np.unique(labels)))
        if score > best_score:
            best_score = score
            best_tau = float(tau)

    labels = fcluster(link, t=best_tau, criterion="distance")
    clusters: Dict[int, List[str]] = {}
    cluster_to_indices: Dict[int, List[int]] = {}
    for row_idx, (entity, label) in enumerate(zip(entity_ids, labels)):
        label = int(label)
        clusters.setdefault(label, []).append(entity)
        cluster_to_indices.setdefault(label, []).append(row_idx)

    # cluster centers (ordered by label)
    cluster_labels = sorted(cluster_to_indices.keys())
    cluster_centers = np.zeros((len(cluster_labels), embeddings.shape[1]), dtype=np.float32)
    for idx, label in enumerate(cluster_labels):
        member_indices = cluster_to_indices[label]
        cluster_centers[idx] = embeddings[member_indices].mean(axis=0)

    logger.info("Selected tau=%.3f with %d clusters", best_tau, len(clusters))
    return SeedHierarchyResult(
        tau_opt=best_tau,
        labels=labels.astype(int),
        clusters=clusters,
        cluster_centers=cluster_centers,
        cluster_labels=cluster_labels,
    )


def compute_neighbor_clusters(
    *,
    cluster_centers: np.ndarray,
    k_neighbors: int = 5,
) -> Dict[int, List[int]]:
    """Compute k nearest neighbor clusters using cosine distance of centers."""

    dist = _cosine_distance_matrix(cluster_centers)
    np.fill_diagonal(dist, np.inf)
    neighbors: Dict[int, List[int]] = {}
    for idx in range(cluster_centers.shape[0]):
        nearest = np.argsort(dist[idx])[:k_neighbors]
        neighbors[int(idx)] = [int(n) for n in nearest]
    return neighbors


def load_seed_hierarchy_artifacts(
    *,
    hierarchy_path: Path,
    cluster_centers_path: Path,
    neighbor_clusters_path: Path,
) -> SeedHierarchyArtifacts:
    if not hierarchy_path.exists():
        raise FileNotFoundError(f"hierarchy_seed.json not found: {hierarchy_path}")
    if not cluster_centers_path.exists():
        raise FileNotFoundError(f"cluster_embeddings.npy not found: {cluster_centers_path}")
    if not neighbor_clusters_path.exists():
        raise FileNotFoundError(f"neighbor_clusters.json not found: {neighbor_clusters_path}")

    with hierarchy_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    entity_ids = tuple(payload["entity_ids"])
    labels = np.asarray(payload["labels"], dtype=np.int64)
    cluster_labels = [int(x) for x in payload.get("cluster_labels", sorted(set(labels.tolist())))]

    centers = np.load(cluster_centers_path)
    with neighbor_clusters_path.open("r", encoding="utf-8") as f:
        neighbor_clusters = {int(k): [int(x) for x in v] for k, v in json.load(f).items()}

    return SeedHierarchyArtifacts(
        entity_ids=entity_ids,
        entity_cluster_labels=labels,
        cluster_labels=cluster_labels,
        cluster_centers=centers,
        neighbor_clusters=neighbor_clusters,
    )
