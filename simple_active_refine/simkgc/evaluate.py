from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from simple_active_refine.util import get_logger

from .data import (
    TripletIndex,
    build_entity_maps,
    collate_batch,
    load_entities_json,
    load_examples_json,
)
from .model import SimKGCModel

logger = get_logger(__name__)


@dataclass
class RankingMetrics:
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float
    mrr: float


def _device_from_model(model: SimKGCModel) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def encode_all_entities(
    *,
    entities: Sequence,
    tokenizer,
    max_num_tokens: int,
    model: SimKGCModel,
    batch_size: int,
) -> torch.Tensor:
    """Encode all entities into a single matrix (N, D)."""

    device = _device_from_model(model)

    def _collate(ent_batch):
        texts = []
        for e in ent_batch:
            if getattr(e, "entity_desc", ""):
                texts.append(f"{e.entity}: {e.entity_desc}")
            else:
                texts.append(e.entity)
        enc = tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=max_num_tokens,
            return_tensors="pt",
        )
        return enc

    dl = DataLoader(
        list(entities),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
    )

    vecs: List[torch.Tensor] = []
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        token_type_ids = batch.get("token_type_ids")
        token_type_ids = token_type_ids.to(device) if token_type_ids is not None else None

        emb = model.encode_entity(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        vecs.append(emb.detach().cpu())

    return torch.cat(vecs, dim=0)


@torch.no_grad()
def evaluate_simkgc_filtered_tail_ranking(
    *,
    artifacts_dir: Path,
    split: str,
    model: SimKGCModel,
    tokenizer,
    max_num_tokens: int,
    batch_size: int,
    hits_k: Sequence[int] = (1, 3, 10),
) -> RankingMetrics:
    """Filtered ranking evaluation for tail prediction.

    For each query (h,r,?) rank all entities as tail candidates.
    """

    device = _device_from_model(model)
    model.eval()

    entities = load_entities_json(artifacts_dir / "entities.json")
    id2ent, ent2idx = build_entity_maps(entities)

    # For filtered evaluation, filter against all known true triples.
    all_triples: List = []
    for sp in ("train", "valid", "test"):
        p = artifacts_dir / f"{sp}.json"
        if p.exists():
            all_triples.extend(load_examples_json(p))
    triplet_index = TripletIndex(all_triples)

    # Pre-encode all entities.
    ent_matrix = encode_all_entities(
        entities=entities,
        tokenizer=tokenizer,
        max_num_tokens=max_num_tokens,
        model=model,
        batch_size=max(8, batch_size),
    )  # (N, D), cpu
    ent_matrix = ent_matrix.to(device)

    examples = load_examples_json(artifacts_dir / f"{split}.json")

    dl = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda b: collate_batch(
            b,
            tokenizer=tokenizer,
            max_num_tokens=max_num_tokens,
            entity_lookup=id2ent,
        ),
    )

    hits = {k: 0 for k in hits_k}
    rr_sum = 0.0
    n = 0

    for batch in dl:
        n_batch = len(batch["batch"])

        hr_input_ids = batch["hr_input_ids"].to(device)
        hr_attention_mask = batch["hr_attention_mask"].to(device) if batch["hr_attention_mask"] is not None else None
        hr_token_type_ids = batch["hr_token_type_ids"].to(device) if batch["hr_token_type_ids"] is not None else None

        hr_emb = model.encode_hr(
            input_ids=hr_input_ids,
            attention_mask=hr_attention_mask,
            token_type_ids=hr_token_type_ids,
        )

        scores = (hr_emb @ ent_matrix.t()) / model.tau  # (B, N)

        # Apply filtered setting: mask all true tails except the target.
        for i, ex in enumerate(batch["batch"]):
            gold_tails = triplet_index.tails(ex.head_id, ex.relation)
            if gold_tails:
                mask_idx = [ent2idx[t] for t in gold_tails if t in ent2idx and t != ex.tail_id]
                if mask_idx:
                    scores[i, torch.tensor(mask_idx, device=device)] = float("-inf")

        # ranks
        for i, ex in enumerate(batch["batch"]):
            gold = ent2idx.get(ex.tail_id)
            if gold is None:
                continue
            # rank = 1 + number of candidates with higher score
            s = scores[i]
            gold_score = s[gold]
            rank = int((s > gold_score).sum().item()) + 1
            rr_sum += 1.0 / rank
            for k in hits_k:
                if rank <= k:
                    hits[k] += 1
            n += 1

    if n == 0:
        return RankingMetrics(hits_at_1=0.0, hits_at_3=0.0, hits_at_10=0.0, mrr=0.0)

    return RankingMetrics(
        hits_at_1=hits.get(1, 0) / n,
        hits_at_3=hits.get(3, 0) / n,
        hits_at_10=hits.get(10, 0) / n,
        mrr=rr_sum / n,
    )
