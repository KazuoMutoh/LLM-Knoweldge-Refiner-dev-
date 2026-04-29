from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import torch
import torch.nn.functional as F
from pykeen.regularizers import Regularizer

from simple_active_refine.util import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class KGFitRegularizerConfig:
    anchor_weight: float = 0.5
    cohesion_weight: float = 0.5
    separation_weight: float = 0.5
    separation_margin: float = 0.2
    profile: bool = False
    profile_every: int = 200


class KGFitRegularizer(Regularizer):
    """Regularizer for KG-FIT anchor and seed hierarchy constraints."""

    def __init__(
        self,
        *,
        anchor_embeddings: torch.FloatTensor,
        entity_cluster_indices: torch.LongTensor,
        cluster_centers: torch.FloatTensor,
        neighbor_clusters: Optional[torch.LongTensor],
        config: KGFitRegularizerConfig,
    ) -> None:
        super().__init__(weight=1.0, apply_only_once=False)
        anchor_norm = F.normalize(anchor_embeddings, dim=-1)
        centers_norm = F.normalize(cluster_centers, dim=-1)
        self.register_buffer("anchor_embeddings", anchor_norm)
        self.register_buffer("entity_cluster_indices", entity_cluster_indices)
        self.register_buffer("cluster_centers", centers_norm)
        if neighbor_clusters is not None:
            self.register_buffer("neighbor_clusters", neighbor_clusters)
        else:
            self.neighbor_clusters = None
        self.config = config
        self._profile_enabled = bool(config.profile)
        self._profile_every = int(config.profile_every)
        self._profile_steps = 0
        self._profile_total = 0.0

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # pragma: no cover - use update_with_indices
        return torch.zeros(1, device=x.device)

    def update_with_indices(self, x: torch.FloatTensor, indices: Optional[torch.LongTensor]) -> None:
        if indices is None:
            return
        if not self.training or not torch.is_grad_enabled():
            return

        start_time = None
        if self._profile_enabled:
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            start_time = time.perf_counter()

        indices = indices.to(self.entity_cluster_indices.device)
        anchors = self.anchor_embeddings[indices]
        cluster_ids = self.entity_cluster_indices[indices]
        centers = self.cluster_centers[cluster_ids]

        x_norm = F.normalize(x, dim=-1)
        anchor_loss = (1.0 - (x_norm * anchors).sum(dim=-1)).mean()
        cohesion_loss = (1.0 - (x_norm * centers).sum(dim=-1)).mean()

        separation_loss = torch.zeros((), device=x.device)
        if self.neighbor_clusters is not None:
            neighbor_ids = self.neighbor_clusters[cluster_ids]
            neighbor_centers = self.cluster_centers[neighbor_ids]
            cosine_sim = torch.einsum("bd,bkd->bk", x_norm, neighbor_centers)
            dist = 1.0 - cosine_sim
            separation_loss = torch.relu(self.config.separation_margin - dist).mean()

        total = (
            self.config.anchor_weight * anchor_loss
            + self.config.cohesion_weight * cohesion_loss
            + self.config.separation_weight * separation_loss
        )
        self.regularization_term = self.regularization_term + total

        if self._profile_enabled:
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            elapsed = time.perf_counter() - start_time
            self._profile_total += elapsed
            self._profile_steps += 1
            if self._profile_steps % self._profile_every == 0:
                avg_ms = (self._profile_total / self._profile_steps) * 1000.0
                last_ms = elapsed * 1000.0
                logger.info(
                    "KGFitRegularizer profile: steps=%d avg_ms=%.3f last_ms=%.3f",
                    self._profile_steps,
                    avg_ms,
                    last_ms,
                )
