from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Union, Dict

import torch
import numpy as np
import os

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline, PipelineResult
from pykeen.models import Model
from pykeen.datasets import get_dataset

Triple = Tuple[str, str, str]


@dataclass
class _SavedMeta:
    dataset_name: Optional[str]
    has_train: bool
    has_valid: bool
    has_test: bool
    normalization: str
    train_score_min: Optional[float]
    train_score_max: Optional[float]


class KnowledgeGraphEmbedding:
    """
    a wrapper for PyKEEN knowledge graph embedding model.
    Usage:
        1) Initialize with trained model directory and triples directory.
        2) Score triples or filter by score range.
    """

    def __init__(self,model_dir: Optional[str] = None):
        """
        Initialize the KnowledgeGraphEmbedding with a trained model.

        Args:
            model_dir: Directory containing the trained model and triples mappings. (created by PipelineResult.save_to_directory())
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_dir is not None:
            
            # lading model and triples
            f_model = os.path.join(model_dir, "trained_model.pkl")
            if not os.path.exists(f_model):
                raise FileNotFoundError(f"trained_model.pkl not found in {model_dir}")
            dir_triples = os.path.join(model_dir, "training_triples")
            if not os.path.exists(dir_triples):
                raise FileNotFoundError(f"training_triples/ not found in {model_dir}. "
                                        f"Provide entity/relation mappings or re-save the run with mappings.")  
            self.triples = TriplesFactory.from_path_binary(dir_triples)
            self.model = torch.load(f_model, map_location=self.device)
            self.model = self.model.to(self.device)
            self.model.eval()

            self._fit_normalization()

        else:
            Exception("model_dir must be provided to load a trained model.")

    @classmethod
    def train_model(self,
                    model: str = "TransE", 
                    dir_triples: Optional[str] = None,
                    dataset_name: Optional[str] = None,
                    split: str = "train",
                    create_inverse_triples: bool = True, 
                    dir_save: Optional[str] = None,
                    **pipeline_kwargs):
        """
        train and return a KnowledgeGraphEmbedding instance.
        Args:
            model_name: Name of the model (e.g., "TransE", "DistMult", etc.)
            dir_triples: Directory containing train.txt, valid.txt, test.txt (at least train.txt).
            dataset_name: Name of a built-in dataset (e.g., "FB15k237", "WN18RR", etc.). If provided, dir_triples is ignored.
            split: Which split to use for training ("train", "valid", "test").
            create_inverse_triples: Whether to create inverse triples for training.
            dir_save: Directory to save the trained model (if None, not saved).
            **pipeline_kwargs: forwarded to pykeen.pipeline.pipeline (e.g., num_epochs=...)
        """

        # --------------------------
        # variables check
        # --------------------------
        if dataset_name is None and dir_triples is None:
            raise ValueError("Either dataset_name or dir_triples must be provided.")
        if dataset_name is not None and dir_triples is not None:
            raise ValueError("Only one of dataset_name or dir_triples should be provided.")
        if dataset_name is not None and split != "train":
            raise ValueError("When using a built-in dataset, split must be 'train'.")
        if dir_triples is not None and not os.path.exists(dir_triples):
            raise FileNotFoundError(f"dir_triples {dir_triples} does not exist.")
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of {'train', 'valid', 'test'}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --------------------------
        # load dataset or triples
        # --------------------------
        if dataset_name is not None:
            dataset = get_dataset(dataset_name, create_inverse_triples=create_inverse_triples)
            train_tf = dataset.training
            valid_tf = dataset.validation
            test_tf = dataset.testing
        else:
            f_train = os.path.join(dir_triples, 'train.txt')
            if os.path.exists(f_train):
                train_tf = TriplesFactory.from_path(path=os.path.join(dir_triples, 'train.txt'), 
                                                    create_inverse_triples=create_inverse_triples)
            else:
                raise FileNotFoundError(f"train.txt not found in {dir_triples}")
            
            f_test = os.path.join(dir_triples, 'test.txt')
            if os.path.exists(f_test):
                test_tf = TriplesFactory.from_path(path=os.path.join(dir_triples, 'test.txt'), 
                                                    entity_to_id=train_tf.entity_to_id,
                                                    relation_to_id=train_tf.relation_to_id)
            else:
                test_tf = None
            
            f_valid = os.path.join(dir_triples, 'valid.txt')
            if os.path.exists(f_valid):
                valid_tf = TriplesFactory.from_path(path=os.path.join(dir_triples, 'valid.txt'), 
                                                    entity_to_id=train_tf.entity_to_id,
                                                    relation_to_id=train_tf.relation_to_id)
            else:
                valid_tf = None

        # --------------------------
        # train model
        # --------------------------
        result = pipeline(
            model=model,
            training=train_tf,
            validation=valid_tf,
            testing=test_tf,
            device=self.device,
            **pipeline_kwargs,
        )
        instance = KnowledgeGraphEmbedding()
        instance.model = result.model.to(self.device)
        instance.triples = train_tf

        # --------------------------
        # preprocess for scoring
        # --------------------------

        # calculate parameters for normalization
        instance.normalization = pipeline_kwargs.get("normalization", "sigmoid")
        instance._train_score_min = None
        instance._train_score_max = None
        
        instance.dir_save = dir_save
        result.save_to_directory(dir_save) if instance.dir_save else None

        return instance

    def _label_to_id(self, triples: List[Triple]) -> Tuple[int, int, int]:
        """
        Convert list of (head, relation, tail) triples to mapped ids using the TriplesFactory.
        Args:
            triples: List of (head, relation, tail)
        Returns:
            Tensor of shape (n, 3) with mapped ids.
        """
        
        h_ids = []
        r_ids = []
        t_ids = []
        for h, r, t in triples:
            if h not in self.triples.entity_to_id or r not in self.triples.relation_to_id or t not in self.triples.entity_to_id:
                raise ValueError(f"Entity or relation not found in the training triples: {(h,r,t)}")
            h_ids.append(self.triples.entity_to_id[h])
            r_ids.append(self.triples.relation_to_id[r])
            t_ids.append(self.triples.entity_to_id[t])

        return torch.tensor(list(zip(h_ids, r_ids, t_ids)), dtype=torch.long, device=self.device)

    def _normalize_scores(self, 
                          scores: torch.Tensor,
                          method='minmax') -> torch.Tensor:
        """
        Normalize scores to [0,1] based on the chosen normalization method.
        Args:
            scores: Tensor of raw scores.
            method: Normalization method ('sigmoid' or 'minmax').
        Returns:
            Tensor of normalized scores in [0,1].
        """
        if method == "sigmoid":
            return torch.sigmoid(scores)
        elif method == "minmax":
            if self._train_score_min is None or self._train_score_max is None:
                raise ValueError("Normalization parameters not set. Cannot use minmax normalization.")
            return (scores - self._train_score_min) / (self._train_score_max - self._train_score_min + 1e-10)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

    def score_triples(self, labeled_triples: List[Triple], 
                      normalize:bool=False, norm_method='minmax') -> List[float]:
        """
        Score a batch of triples.
        Args:
            labeled_triples: List of (head, relation, tail)
            normalize: Whether to normalize scores to [0,1].
            norm_method: Normalization method ('sigmoid' or 'minmax').
        Returns:
            List of scores in [0,1] (normalized).
        """
        triples = self._label_to_id(labeled_triples)

        with torch.no_grad():
            scores = self.model.score_hrt(triples)
        
        if normalize:
            scores = self._normalize_scores(scores, method=norm_method)

        return [s[0] for s in scores.tolist()]

    def score_triple(self, labeled_triple: Triple) -> float:
        """
        Score a single triple.
        Args:
            labeled_triple: (head, relation, tail)
        Returns:
            Score in [0,1] (normalized).
        """
        return self.score_triples([labeled_triple])[0]

    def filter_triples_by_score(self,
                                labeled_triples: List[Triple],
                                score_min: float = None,
                                score_max: float = None,
                                upper_percentile: Optional[float] = None,
                                lower_percentile: Optional[float] = None,
                                return_with_score: bool = False) -> List[Union[Triple, Tuple[Triple, float]]]:
        """
        Filter triples based on their scores.
        Args:
            score_min: Minimum score (inclusive).
            score_max: Maximum score (inclusive).
            labeled_triples: List of (head, relation, tail)
            return_with_score: If True, return list of (triple, score) tuples. If False, return list of triples.
            batch_size: Batch size for scoring.
        Returns:
            List of triples (and optionally their scores) within the specified score range.
        """

        # socre_minとscore_maxがNoneの場合、パーセンタイルで指定されている場合は計算
        if score_min is None and lower_percentile is not None:
            all_scores = self.score_triples(labeled_triples)
            score_min = float(np.percentile(all_scores, lower_percentile))
        if score_max is None and upper_percentile is not None:
            all_scores = self.score_triples(labeled_triples)
            score_max = float(np.percentile(all_scores, upper_percentile)) 
        if score_min is None:
            score_min = -float('inf')
        if score_max is None:
            score_max = float('inf')
        if score_min > score_max:
            raise ValueError("score_min cannot be greater than score_max.")
        if not labeled_triples:
            return []
    
        scores = self.score_triples(labeled_triples)
        results = []
        for triple, score in zip(labeled_triples, scores):
            if score_min <= score <= score_max:
                results.append((triple, score) if return_with_score else triple)
        return results
    
    def get_labeled_triples(self) -> List[Triple]:
        """
        Get all labeled triples from the training data.
        Returns:
            List of (head, relation, tail)
        """
        return self.triples.triples.tolist()
    
    def _fit_normalization(self):
        """
        Fit normalization parameters based on training triples scores.
        """
        with torch.no_grad():
            scores = self.model.score_hrt(self.triples.mapped_triples.to(self.device)).cpu().numpy()
        self._train_score_min = float(np.min(scores))
        self._train_score_max = float(np.max(scores))
    