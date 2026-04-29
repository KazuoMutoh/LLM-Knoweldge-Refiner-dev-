from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SimKGCExample:
    """A single (h, r, t) training/evaluation example."""

    head_id: str
    relation: str
    tail_id: str
    head: str
    tail: str


@dataclass(frozen=True)
class EntityExample:
    """Entity text metadata used for building input strings."""

    entity_id: str
    entity: str
    entity_desc: str = ""


class TripletIndex:
    """Index for filtered evaluation and false-negative masking.

    Stores a mapping from (head_id, relation) -> set(tail_id).
    """

    def __init__(self, examples: Iterable[SimKGCExample]):
        hr2tails: Dict[Tuple[str, str], set[str]] = {}
        for ex in examples:
            hr2tails.setdefault((ex.head_id, ex.relation), set()).add(ex.tail_id)
        self._hr2tails = hr2tails

    def tails(self, head_id: str, relation: str) -> set[str]:
        return self._hr2tails.get((head_id, relation), set())


def load_examples_json(path: Path) -> List[SimKGCExample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[SimKGCExample] = []
    for obj in data:
        out.append(
            SimKGCExample(
                head_id=str(obj["head_id"]),
                relation=str(obj["relation"]),
                tail_id=str(obj["tail_id"]),
                head=str(obj.get("head", obj["head_id"])),
                tail=str(obj.get("tail", obj["tail_id"])),
            )
        )
    return out


def load_entities_json(path: Path) -> List[EntityExample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[EntityExample] = []
    for obj in data:
        out.append(
            EntityExample(
                entity_id=str(obj["entity_id"]),
                entity=str(obj.get("entity", obj["entity_id"])),
                entity_desc=str(obj.get("entity_desc", "")),
            )
        )
    return out


def build_entity_maps(entities: Sequence[EntityExample]) -> tuple[Dict[str, EntityExample], Dict[str, int]]:
    id2ent = {e.entity_id: e for e in entities}
    ent2idx = {e.entity_id: i for i, e in enumerate(entities)}
    return id2ent, ent2idx


def concat_name_desc(name: str, desc: str) -> str:
    name = name or ""
    desc = desc or ""
    desc = desc.strip()
    if not desc:
        return name
    if desc.startswith(name):
        desc = desc[len(name) :].strip()
    return f"{name}: {desc}" if desc else name


def make_hr_text(*, head: EntityExample, relation: str) -> tuple[str, str]:
    """Return (text, text_pair) for encoding (h,r).

    We encode (h,r) as `text=head_text` and `text_pair=relation`.
    """

    head_text = concat_name_desc(head.entity, head.entity_desc)
    return head_text, relation


def make_entity_text(entity: EntityExample) -> str:
    return concat_name_desc(entity.entity, entity.entity_desc)


def collate_batch(
    batch: Sequence[SimKGCExample],
    *,
    tokenizer,
    max_num_tokens: int,
    entity_lookup: Dict[str, EntityExample],
) -> dict:
    """Collate a batch into model inputs.

    Args:
        batch: Examples.
        tokenizer: Tokenizer object with HuggingFace-like call signature.
        max_num_tokens: Max tokens.
        entity_lookup: entity_id -> EntityExample.

    Returns:
        Dict of tensors + batch metadata.
    """

    head_texts: List[str] = []
    rel_texts: List[str] = []
    tail_texts: List[str] = []
    head_only_texts: List[str] = []

    for ex in batch:
        head_ex = entity_lookup.get(ex.head_id, EntityExample(ex.head_id, ex.head, ""))
        tail_ex = entity_lookup.get(ex.tail_id, EntityExample(ex.tail_id, ex.tail, ""))

        head_text, rel_text = make_hr_text(head=head_ex, relation=ex.relation)
        head_texts.append(head_text)
        rel_texts.append(rel_text)
        tail_texts.append(make_entity_text(tail_ex))
        head_only_texts.append(make_entity_text(head_ex))

    hr = tokenizer(
        text=head_texts,
        text_pair=rel_texts,
        padding=True,
        truncation=True,
        max_length=max_num_tokens,
        return_tensors="pt",
    )
    tail = tokenizer(
        text=tail_texts,
        padding=True,
        truncation=True,
        max_length=max_num_tokens,
        return_tensors="pt",
    )
    head_only = tokenizer(
        text=head_only_texts,
        padding=True,
        truncation=True,
        max_length=max_num_tokens,
        return_tensors="pt",
    )

    return {
        "hr_input_ids": hr["input_ids"],
        "hr_attention_mask": hr.get("attention_mask"),
        "hr_token_type_ids": hr.get("token_type_ids"),
        "tail_input_ids": tail["input_ids"],
        "tail_attention_mask": tail.get("attention_mask"),
        "tail_token_type_ids": tail.get("token_type_ids"),
        "head_input_ids": head_only["input_ids"],
        "head_attention_mask": head_only.get("attention_mask"),
        "head_token_type_ids": head_only.get("token_type_ids"),
        "batch": list(batch),
    }


def build_in_batch_mask(batch: Sequence[SimKGCExample], triplets: TripletIndex) -> torch.Tensor:
    """Build in-batch mask to filter false negatives.

    Returns:
        Bool tensor of shape (B, B) where True means allowed.
    """

    bsz = len(batch)
    mask = torch.ones((bsz, bsz), dtype=torch.bool)
    for i, ex in enumerate(batch):
        gold = triplets.tails(ex.head_id, ex.relation)
        if not gold:
            continue
        for j, cand in enumerate(batch):
            if i == j:
                continue
            if cand.tail_id in gold:
                mask[i, j] = False
    return mask


def build_self_negative_mask(batch: Sequence[SimKGCExample], triplets: TripletIndex) -> torch.Tensor:
    """Mask for self-negative term.

    True means the self-negative is allowed.
    """

    allowed = torch.zeros((len(batch),), dtype=torch.bool)
    for i, ex in enumerate(batch):
        if ex.head_id == ex.tail_id:
            continue
        gold = triplets.tails(ex.head_id, ex.relation)
        if ex.head_id in gold:
            continue
        allowed[i] = True
    return allowed
