from __future__ import annotations

from typing import Optional

import torch
from pykeen.nn import Embedding

from simple_active_refine.kgfit_regularizer import KGFitRegularizer


class KGFitEntityEmbedding(Embedding):
    """Embedding representation that updates KG-FIT regularizer with indices."""

    def __init__(self, *, kgfit_regularizer: KGFitRegularizer, **kwargs) -> None:
        kwargs.setdefault("unique", True)
        super().__init__(**kwargs)
        self.kgfit_regularizer = kgfit_regularizer
        # register as submodule to be collected by ERModel.collect_regularization_term
        self.add_module("kgfit_regularizer", kgfit_regularizer)

    def forward(self, indices: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        inverse = None
        if indices is not None and self.unique:
            indices, inverse = indices.unique(return_inverse=True)
        x = self._plain_forward(indices=indices)
        if self.normalizer is not None:
            x = self.normalizer(x)
        if inverse is not None:
            x = x[inverse]
        if indices is not None:
            if inverse is not None:
                indices = indices[inverse]
            self.kgfit_regularizer.update_with_indices(x, indices)
        if self.dropout is not None:
            x = self.dropout(x)
        return x
