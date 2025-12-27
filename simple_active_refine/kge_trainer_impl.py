"""Concrete KGE trainer implementation."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Dict

from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.pipeline import BaseKGETrainer, KGETrainingContext, KGETrainingResult
from simple_active_refine.data_manager import IterationDataManager
from simple_active_refine.util import get_logger

logger = get_logger(__name__)


class FinalKGETrainer(BaseKGETrainer):
    """Train a KGE on the refined KG and report rank-based metrics."""

    def __init__(self, data_manager: IterationDataManager, embedding_config: Dict) -> None:
        self.data_manager = data_manager
        self.embedding_config = embedding_config

    def train_and_evaluate(self, context: KGETrainingContext) -> KGETrainingResult:
        dir_name = os.path.basename(context.output_dir) if context.output_dir else "final"
        dataset_dir = self.data_manager.write_custom(dir_name, context.kg)
        output_dir = context.output_dir or dataset_dir

        config = deepcopy(self.embedding_config)
        config["dir_triples"] = dataset_dir
        config["dir_save"] = output_dir
        kge = KnowledgeGraphEmbedding.train_model(**config)

        metrics = kge.evaluate()
        diagnostics = {"output_dir": output_dir}
        dump_path = os.path.join(output_dir, "kge_training_io.json")
        try:
            payload = {
                "input": {
                    "dataset_dir": dataset_dir,
                    "output_dir": output_dir,
                },
                "output": {
                    "metrics": metrics,
                    "model_path": output_dir,
                },
            }
            with open(dump_path, "w", encoding="utf-8") as fout:
                json.dump(payload, fout, ensure_ascii=False, indent=2)
        except Exception as err:
            logger.warning("[v3] Failed to dump KGE training IO: %s", err)
        return KGETrainingResult(
            hits_at_1=metrics.get("hits_at_1", 0.0),
            hits_at_3=metrics.get("hits_at_3", 0.0),
            hits_at_10=metrics.get("hits_at_10", 0.0),
            mrr=metrics.get("mean_reciprocal_rank", 0.0),
            model_path=output_dir,
            diagnostics=diagnostics,
        )
