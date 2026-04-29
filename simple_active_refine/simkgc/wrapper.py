from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from simple_active_refine.util import get_logger

from .config import SimKGCConfig
from .dummy import DummyEncoder, DummyTokenizer
from .evaluate import RankingMetrics, evaluate_simkgc_filtered_tail_ranking
from .model import SimKGCModel
from .train import train_simkgc

logger = get_logger(__name__)


class SimKGCWrapper:
    """Backend wrapper to integrate SimKGC with this repo."""

    def __init__(
        self,
        *,
        config: SimKGCConfig,
        artifacts_dir: Path,
        output_dir: Path,
    ):
        self.config = config
        self.artifacts_dir = artifacts_dir
        self.output_dir = output_dir
        self.model: Optional[SimKGCModel] = None
        self.tokenizer = None

    def _build_tokenizer_and_model(self) -> tuple[object, SimKGCModel]:
        # For now, default to dummy offline encoder to keep tests stable.
        if self.config.pretrained_model == "__dummy__":
            tokenizer = DummyTokenizer()
            encoder = DummyEncoder(hidden_size=self.config.embedding_dim)
            model = SimKGCModel(
                hr_encoder=encoder,
                entity_encoder=encoder,
                temperature=self.config.temperature,
                learnable_temperature=self.config.learnable_temperature,
            )
            return tokenizer, model

        # Lazy import to avoid heavy deps for tests.
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model)
        hr_encoder = AutoModel.from_pretrained(self.config.pretrained_model)
        ent_encoder = AutoModel.from_pretrained(self.config.pretrained_model)

        # Wrap HF models to return pooled embeddings.
        model = SimKGCModel(
            hr_encoder=_HFMeanPooler(hr_encoder),
            entity_encoder=_HFMeanPooler(ent_encoder),
            temperature=self.config.temperature,
            learnable_temperature=self.config.learnable_temperature,
        )
        return tokenizer, model

    def train(self) -> Dict:
        self.tokenizer, self.model = self._build_tokenizer_and_model()
        summary = train_simkgc(
            artifacts_dir=self.artifacts_dir,
            config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=self.output_dir,
        )
        return summary

    def load(self, checkpoint_path: Optional[Path] = None) -> None:
        if checkpoint_path is None:
            checkpoint_path = self.output_dir / "simkgc.pt"
        self.tokenizer, self.model = self._build_tokenizer_and_model()
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt["state_dict"], strict=True)

    @torch.no_grad()
    def score_triples(
        self,
        *,
        triples: Sequence[Tuple[str, str, str]],
        entity_text: Dict[str, str],
        relation_text: Optional[Dict[str, str]] = None,
        normalize_0_1: bool = True,
    ) -> np.ndarray:
        """Score (h,r,t) triples.

        Args:
            triples: List of (head_id, relation, tail_id).
            entity_text: Map of entity_id -> text.
            relation_text: Optional map of relation -> text.
            normalize_0_1: If True, map cosine/inner product scores to 0-1.
        """

        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None

        device = next(self.model.parameters()).device
        self.model.eval()

        # Batch encode hr and tail.
        head_texts = []
        rel_texts = []
        tail_texts = []
        for h, r, t in triples:
            head_texts.append(entity_text.get(h, h))
            rel_texts.append((relation_text or {}).get(r, r))
            tail_texts.append(entity_text.get(t, t))

        hr = self.tokenizer(
            text=head_texts,
            text_pair=rel_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_num_tokens,
            return_tensors="pt",
        )
        tail = self.tokenizer(
            text=tail_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_num_tokens,
            return_tensors="pt",
        )

        hr_emb = self.model.encode_hr(
            input_ids=hr["input_ids"].to(device),
            attention_mask=hr.get("attention_mask").to(device) if hr.get("attention_mask") is not None else None,
            token_type_ids=hr.get("token_type_ids").to(device) if hr.get("token_type_ids") is not None else None,
        )
        tail_emb = self.model.encode_entity(
            input_ids=tail["input_ids"].to(device),
            attention_mask=tail.get("attention_mask").to(device) if tail.get("attention_mask") is not None else None,
            token_type_ids=tail.get("token_type_ids").to(device) if tail.get("token_type_ids") is not None else None,
        )

        scores = (hr_emb * tail_emb).sum(dim=1) / self.model.tau
        scores = scores.detach().cpu().numpy()

        if normalize_0_1:
            # Normalize to 0-1 via logistic.
            scores = 1.0 / (1.0 + np.exp(-scores))

        return scores

    def evaluate(self, split: str = "test") -> RankingMetrics:
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None
        return evaluate_simkgc_filtered_tail_ranking(
            artifacts_dir=self.artifacts_dir,
            split=split,
            model=self.model,
            tokenizer=self.tokenizer,
            max_num_tokens=self.config.max_num_tokens,
            batch_size=self.config.eval_batch_size,
        )


class _HFMeanPooler(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        x = out.last_hidden_state
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (x * mask).sum(dim=1) / denom
        return torch.nn.functional.normalize(pooled, p=2, dim=-1)
