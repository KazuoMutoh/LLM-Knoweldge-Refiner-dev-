from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
from pykeen.models import Model


@dataclass
class Metrics:
    mrr: float
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float


def evaluate(model: Model, factory_train: TriplesFactory,
             factory_test: Optional[TriplesFactory]) -> Optional[Metrics]:
    """Compute filtered ranking metrics if test factory is provided.

    Args:
        model: Trained model.
        factory_train: Training factory (for filtering known triples).
        factory_test: Test factory.

    Returns:
        Metrics or None if no test set.
    """
    if factory_test is None:
        return None
    evaluator = RankBasedEvaluator()
    res = evaluator.evaluate(model=model, mapped_triples=factory_test.mapped_triples,
                             additional_filter_triples=[factory_train.mapped_triples])
    return Metrics(
        mrr=float(res.get_metric("mrr")),
        hits_at_1=float(res.get_metric("hits@1")),
        hits_at_3=float(res.get_metric("hits@3")),
        hits_at_10=float(res.get_metric("hits@10")),
    )