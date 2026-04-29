from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SimKGCLossConfig:
    additive_margin: float = 0.0
    use_in_batch_negatives: bool = True
    use_self_negative: bool = False
    self_negative_weight: float = 1.0


def simkgc_info_nce_loss(
    scores: torch.Tensor,
    *,
    gold_index: torch.Tensor,
    additive_margin: float = 0.0,
    in_batch_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """InfoNCE loss with optional additive margin and false-negative masking.

    Args:
        scores: (B, N) scores (higher is better).
        gold_index: (B,) target indices into N.
        additive_margin: Subtracted from positive score.
        in_batch_mask: Optional bool mask (B, N). False entries are masked out.

    Returns:
        Scalar loss.
    """

    if in_batch_mask is not None:
        scores = scores.masked_fill(~in_batch_mask.to(scores.device), float("-inf"))

    batch_idx = torch.arange(scores.size(0), device=scores.device)
    pos = scores[batch_idx, gold_index] - float(additive_margin)

    # logsumexp over candidates.
    denom = torch.logsumexp(scores, dim=1)
    loss = -(pos - denom)
    return loss.mean()


def simkgc_self_negative_loss(
    hr_emb: torch.Tensor,
    head_emb: torch.Tensor,
    *,
    tau: torch.Tensor,
    allowed_mask: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Self-negative term: treat head as negative tail when allowed."""

    if allowed_mask.numel() == 0:
        return torch.tensor(0.0, device=hr_emb.device)

    allowed = allowed_mask.to(hr_emb.device)
    if allowed.sum().item() == 0:
        return torch.tensor(0.0, device=hr_emb.device)

    # score for head as negative: (hr · head) / tau
    s = (hr_emb * head_emb).sum(dim=1) / tau
    # maximize log(1 - sigmoid(s)) == minimize -log(sigmoid(-s))
    loss = F.softplus(s)  # == -log(sigmoid(-s))
    loss = loss[allowed]
    return loss.mean() * float(weight)
