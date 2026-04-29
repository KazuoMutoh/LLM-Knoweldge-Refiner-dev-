from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from simple_active_refine.util import get_logger

from .config import SimKGCConfig
from .data import (
    TripletIndex,
    build_in_batch_mask,
    build_self_negative_mask,
    collate_batch,
    load_entities_json,
    load_examples_json,
)
from .losses import (
    SimKGCLossConfig,
    simkgc_info_nce_loss,
    simkgc_self_negative_loss,
)
from .model import SimKGCModel

logger = get_logger(__name__)


def json_dumps_pretty(obj: Dict) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def _device_from_config(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_simkgc(
    *,
    artifacts_dir: Path,
    config: SimKGCConfig,
    model: SimKGCModel,
    tokenizer,
    output_dir: Path,
) -> Dict:
    """Train SimKGC on prepared artifacts.

    Artifacts expected:
        - train.json
        - valid.json (optional)
        - entities.json

    Saves:
        - output_dir/simkgc.pt
        - output_dir/simkgc_config.json

    Returns:
        Training summary dict.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_config(config.device)
    model = model.to(device)
    model.train()

    train_path = artifacts_dir / "train.json"
    entities_path = artifacts_dir / "entities.json"
    train_examples = load_examples_json(train_path)
    entities = load_entities_json(entities_path)
    entity_lookup = {e.entity_id: e for e in entities}
    triplet_index = TripletIndex(train_examples)

    dl = DataLoader(
        train_examples,
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda b: collate_batch(
            b,
            tokenizer=tokenizer,
            max_num_tokens=config.max_num_tokens,
            entity_lookup=entity_lookup,
        ),
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    loss_cfg = SimKGCLossConfig(
        additive_margin=config.additive_margin,
        use_in_batch_negatives=True,
        use_self_negative=config.use_self_negative,
        self_negative_weight=config.self_negative_weight,
    )

    global_step = 0
    running_loss = 0.0

    for epoch in range(config.num_epochs):
        for batch in dl:
            global_step += 1
            hr_input_ids = batch["hr_input_ids"].to(device)
            hr_attention_mask = batch["hr_attention_mask"].to(device) if batch["hr_attention_mask"] is not None else None
            hr_token_type_ids = batch["hr_token_type_ids"].to(device) if batch["hr_token_type_ids"] is not None else None

            tail_input_ids = batch["tail_input_ids"].to(device)
            tail_attention_mask = (
                batch["tail_attention_mask"].to(device) if batch["tail_attention_mask"] is not None else None
            )
            tail_token_type_ids = (
                batch["tail_token_type_ids"].to(device) if batch["tail_token_type_ids"] is not None else None
            )

            head_input_ids = batch["head_input_ids"].to(device)
            head_attention_mask = (
                batch["head_attention_mask"].to(device) if batch["head_attention_mask"] is not None else None
            )
            head_token_type_ids = (
                batch["head_token_type_ids"].to(device) if batch["head_token_type_ids"] is not None else None
            )

            out = model(
                hr_input_ids=hr_input_ids,
                hr_attention_mask=hr_attention_mask,
                hr_token_type_ids=hr_token_type_ids,
                tail_input_ids=tail_input_ids,
                tail_attention_mask=tail_attention_mask,
                tail_token_type_ids=tail_token_type_ids,
                head_input_ids=head_input_ids,
                head_attention_mask=head_attention_mask,
                head_token_type_ids=head_token_type_ids,
            )

            scores = model.score_matrix(out.hr_emb, out.tail_emb)  # (B, B)
            gold_index = torch.arange(scores.size(0), device=scores.device)

            in_batch_mask = None
            if loss_cfg.use_in_batch_negatives:
                in_batch_mask = build_in_batch_mask(batch["batch"], triplet_index).to(scores.device)

            loss = simkgc_info_nce_loss(
                scores,
                gold_index=gold_index,
                additive_margin=loss_cfg.additive_margin,
                in_batch_mask=in_batch_mask,
            )

            if loss_cfg.use_self_negative:
                allowed = build_self_negative_mask(batch["batch"], triplet_index)
                loss = loss + simkgc_self_negative_loss(
                    out.hr_emb,
                    out.head_emb,
                    tau=model.tau.detach(),
                    allowed_mask=allowed,
                    weight=loss_cfg.self_negative_weight,
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running_loss += float(loss.item())

        logger.info(
            "[simkgc] epoch=%s loss=%.4f", epoch + 1, running_loss / max(1, len(dl))
        )

    ckpt = {
        "state_dict": model.state_dict(),
        "simkgc_config": asdict(config),
    }
    torch.save(ckpt, output_dir / "simkgc.pt")

    # Also save json for inspection
    (output_dir / "simkgc_config.json").write_text(
        json_dumps_pretty(asdict(config)), encoding="utf-8"
    )

    return {
        "epochs": config.num_epochs,
        "steps": global_step,
        "avg_loss": running_loss / max(1, global_step),
        "checkpoint": str(output_dir / "simkgc.pt"),
    }
