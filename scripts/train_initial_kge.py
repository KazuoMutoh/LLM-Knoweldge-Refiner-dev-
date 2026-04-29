#!/usr/bin/env python3
"""Train initial KGE model for experiments.

This script trains a knowledge graph embedding model using the specified
dataset and configuration, and saves it to the output directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.util import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train initial KGE model")
    parser.add_argument("--dir_triples", required=True, help="Directory containing train.txt, valid.txt, test.txt")
    parser.add_argument("--output_dir", required=True, help="Output directory for the trained model")
    parser.add_argument("--embedding_config", required=True, help="Path to embedding configuration JSON file")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation after training")
    
    args = parser.parse_args()
    
    # Load embedding config
    logger.info(f"Loading embedding config from: {args.embedding_config}")
    with open(args.embedding_config, "r") as f:
        config = json.load(f)
    
    # Extract parameters that go directly to train_model
    model = config.pop("model", "TransE")
    embedding_backend = config.pop("embedding_backend", "pykeen")
    kgfit_config = config.pop("kgfit", None)
    
    # Override parameters
    logger.info(f"Training KGE model:")
    logger.info(f"  Model: {model}")
    logger.info(f"  Dataset directory: {args.dir_triples}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Number of epochs: {args.num_epochs}")
    logger.info(f"  Pipeline kwargs: {config}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    logger.info("Starting model training...")
    kge = KnowledgeGraphEmbedding.train_model(
        model=model,
        dir_triples=args.dir_triples,
        dir_save=args.output_dir,
        embedding_backend=embedding_backend,
        kgfit_config=kgfit_config,
        skip_evaluation=args.skip_evaluation,
        epochs=args.num_epochs,
        **config
    )
    logger.info("Model training completed")
    
    # Evaluate model (optional: dataset may not have valid/test splits)
    logger.info("Evaluating model...")
    metrics = {}
    if args.skip_evaluation:
        logger.info("Evaluation skipped: skip_evaluation flag is set")
        metrics = {"evaluation_skipped": True, "reason": "skip_evaluation flag is set"}
    else:
        try:
            metrics = kge.evaluate()
            logger.info(f"Evaluation metrics: {metrics}")
        except ValueError as err:
            logger.warning("Evaluation skipped: %s", err)
            metrics = {"evaluation_skipped": True, "reason": str(err)}

    # Save metrics (always write so downstream scripts can detect completion)
    metrics_path = os.path.join(args.output_dir, "initial_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
