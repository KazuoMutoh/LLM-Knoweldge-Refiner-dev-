from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


def _stable_hash(text: str) -> int:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


@dataclass
class DummyBatchEncoding:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor


class DummyTokenizer:
    """Offline tokenizer compatible with HuggingFace tokenizers call signature."""

    def __init__(self, vocab_size: int = 30522):
        self.vocab_size = vocab_size

    def __call__(
        self,
        *,
        text: List[str],
        text_pair: Optional[List[str]] = None,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 64,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        if text_pair is None:
            text_pair = [""] * len(text)
        assert len(text) == len(text_pair)

        batch_ids: List[List[int]] = []
        for t, p in zip(text, text_pair):
            # Deterministic pseudo-token IDs based on hashes.
            ids = [101]  # [CLS]
            ids.append((_stable_hash(t) % (self.vocab_size - 103)) + 103)
            ids.append((_stable_hash(p) % (self.vocab_size - 103)) + 103)
            ids.append(102)  # [SEP]
            ids = ids[:max_length]
            batch_ids.append(ids)

        max_len = max(len(x) for x in batch_ids)
        if padding:
            max_len = max(max_len, min(max_length, max_len))

        input_ids = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
        token_type_ids = torch.zeros((len(batch_ids), max_len), dtype=torch.long)

        for i, ids in enumerate(batch_ids):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
            # token_type_ids stays zero

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }


class DummyEncoder(nn.Module):
    """Offline encoder that maps input_ids to normalized embeddings."""

    def __init__(self, hidden_size: int = 128, vocab_size: int = 30522):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.emb(input_ids)  # (B, T, H)
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (x * mask).sum(dim=1) / denom
        return torch.nn.functional.normalize(pooled, p=2, dim=-1)
