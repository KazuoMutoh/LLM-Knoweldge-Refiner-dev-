from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SimKGCOutput:
    hr_emb: torch.Tensor
    tail_emb: torch.Tensor
    head_emb: torch.Tensor


class SimKGCModel(nn.Module):
    """Minimal SimKGC bi-encoder.

    Encodes (h,r) pairs and entities into a shared embedding space.
    """

    def __init__(
        self,
        *,
        hr_encoder: nn.Module,
        entity_encoder: nn.Module,
        temperature: float = 0.05,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.hr_encoder = hr_encoder
        self.entity_encoder = entity_encoder
        init_tau = torch.tensor(float(temperature))
        if learnable_temperature:
            self.log_tau = nn.Parameter(init_tau.log())
        else:
            self.register_buffer("log_tau", init_tau.log())

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def encode_hr(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.hr_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def encode_entity(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.entity_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def forward(
        self,
        *,
        hr_input_ids: torch.Tensor,
        hr_attention_mask: Optional[torch.Tensor] = None,
        hr_token_type_ids: Optional[torch.Tensor] = None,
        tail_input_ids: torch.Tensor,
        tail_attention_mask: Optional[torch.Tensor] = None,
        tail_token_type_ids: Optional[torch.Tensor] = None,
        head_input_ids: Optional[torch.Tensor] = None,
        head_attention_mask: Optional[torch.Tensor] = None,
        head_token_type_ids: Optional[torch.Tensor] = None,
    ) -> SimKGCOutput:
        hr_emb = self.encode_hr(
            input_ids=hr_input_ids,
            attention_mask=hr_attention_mask,
            token_type_ids=hr_token_type_ids,
        )
        tail_emb = self.encode_entity(
            input_ids=tail_input_ids,
            attention_mask=tail_attention_mask,
            token_type_ids=tail_token_type_ids,
        )
        if head_input_ids is None:
            head_emb = torch.empty((hr_emb.size(0), hr_emb.size(1)), device=hr_emb.device, dtype=hr_emb.dtype)
        else:
            head_emb = self.encode_entity(
                input_ids=head_input_ids,
                attention_mask=head_attention_mask,
                token_type_ids=head_token_type_ids,
            )
        return SimKGCOutput(hr_emb=hr_emb, tail_emb=tail_emb, head_emb=head_emb)

    def score_matrix(self, hr_emb: torch.Tensor, tail_emb: torch.Tensor) -> torch.Tensor:
        """Compute scores for all tails in batch.

        Returns:
            (B, N) scores = (hr_emb @ tail_emb.T) / tau.
        """

        return (hr_emb @ tail_emb.t()) / self.tau
