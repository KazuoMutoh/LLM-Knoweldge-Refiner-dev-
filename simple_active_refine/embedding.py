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
from pykeen.evaluation import RankBasedEvaluator
from pykeen.nn import Embedding
from pykeen.nn import PairREInteraction, TransEInteraction
from pykeen.nn.init import PretrainedInitializer
from pykeen.models import ERModel

from simple_active_refine.util import get_logger
from simple_active_refine.kgfit import (
    KGFitEmbeddingConfig,
    KGFitEmbeddingError,
    load_kgfit_entity_embeddings,
    resolve_kgfit_paths,
)
from simple_active_refine.kgfit_hierarchy import load_seed_hierarchy_artifacts
from simple_active_refine.kgfit_regularizer import KGFitRegularizer, KGFitRegularizerConfig
from simple_active_refine.kgfit_representation import KGFitEntityEmbedding

from simple_active_refine.simkgc.artifacts import prepare_simkgc_artifacts
from simple_active_refine.simkgc.config import SimKGCConfig
from simple_active_refine.simkgc.data import load_entities_json
from simple_active_refine.simkgc.wrapper import SimKGCWrapper

logger = get_logger(__name__)

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

    def __init__(self, model_dir: Optional[str] = None, fit_normalization: bool = True):
        """
        Initialize the KnowledgeGraphEmbedding with a trained model.

        Args:
            model_dir: Directory containing the trained model and triples mappings. (created by PipelineResult.save_to_directory())
            fit_normalization: Whether to fit normalization parameters on training scores.
        """
        self.device = self._get_safe_device()
        self.dir_save = model_dir  # Store the model directory
        self.embedding_backend: str = "pykeen"
        self._simkgc: Optional[SimKGCWrapper] = None
        self._simkgc_entity_text: Dict[str, str] = {}
        self._simkgc_relation_text: Dict[str, str] = {}
        
        if model_dir is not None:

            # ---------------------------------
            # SimKGC backend: load lightweight checkpoint + artifacts
            # ---------------------------------
            simkgc_ckpt = os.path.join(model_dir, "simkgc.pt")
            if os.path.exists(simkgc_ckpt):
                self.embedding_backend = "simkgc"
                artifacts_dir = Path(model_dir) / "simkgc_artifacts"

                # Load config
                config_path = Path(model_dir) / "simkgc_config.json"
                if config_path.exists():
                    import json

                    cfg_payload = json.loads(config_path.read_text(encoding="utf-8"))
                    cfg = SimKGCConfig.from_dict(cfg_payload)
                else:
                    cfg = SimKGCConfig(pretrained_model="__dummy__")

                self._simkgc = SimKGCWrapper(
                    config=cfg,
                    artifacts_dir=artifacts_dir,
                    output_dir=Path(model_dir),
                )
                self._simkgc.load(Path(simkgc_ckpt))

                # Load entity/relation texts
                try:
                    entities = load_entities_json(artifacts_dir / "entities.json")
                    for e in entities:
                        if e.entity_desc:
                            self._simkgc_entity_text[e.entity_id] = f"{e.entity}: {e.entity_desc}"
                        else:
                            self._simkgc_entity_text[e.entity_id] = e.entity
                except Exception:
                    self._simkgc_entity_text = {}

                rel2text = artifacts_dir / "relation2text.txt"
                if rel2text.exists():
                    for line in rel2text.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            self._simkgc_relation_text[parts[0]] = parts[1]

                # SimKGC produces already-normalized scores via sigmoid/logistic in wrapper
                self._train_score_min = None
                self._train_score_max = None
                return
            
            # lading model and triples
            # Allow pickle to deserialize pathlib objects saved by older PyTorch
            if hasattr(torch.serialization, "add_safe_globals"):
                try:
                    torch.serialization.add_safe_globals([Path])
                except Exception:
                    pass

            f_model = os.path.join(model_dir, "trained_model.pkl")
            if not os.path.exists(f_model):
                raise FileNotFoundError(f"trained_model.pkl not found in {model_dir}")
            dir_triples = os.path.join(model_dir, "training_triples")
            if not os.path.exists(dir_triples):
                raise FileNotFoundError(f"training_triples/ not found in {model_dir}. "
                                        f"Provide entity/relation mappings or re-save the run with mappings.")

            # Force torch.load to allow full pickle objects during triples factory load (PyTorch 2.6+ default changed)
            orig_load = torch.load
            def _safe_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return orig_load(*args, **kwargs)
            torch.load = _safe_load
            try:
                self.triples = TriplesFactory.from_path_binary(dir_triples)
            finally:
                torch.load = orig_load
            try:
                # PyTorch 2.6+ defaults to weights_only=True; explicitly allow full load for trusted checkpoints
                # Load on CPU first to avoid CUDA/NVML initialization during deserialization.
                self.model = torch.load(f_model, map_location="cpu", weights_only=False)
            except TypeError:
                # For older torch without weights_only parameter
                self.model = torch.load(f_model, map_location="cpu")
            self.model = self.model.to(self.device)
            self.model.eval()

            if fit_normalization:
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
                    embedding_backend: str = "pykeen",
                    kgfit_config: Optional[dict] = None,
                    skip_evaluation: bool = False,
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

        self.device = KnowledgeGraphEmbedding._get_safe_device()

        # --------------------------
        # PyKEEN pipeline evaluation defaults
        # --------------------------
        # PyKEEN's rank-based evaluation can allocate very large intermediate
        # tensors (especially for high-dimensional models such as KG-FIT + PairRE).
        # When running on CPU, this can easily OOM unless slice_size is set.
        #
        # Additionally, torch_max_mem (if installed) may try to auto-maximize
        # evaluation parameters on non-CUDA devices, which can be unsafe.
        # Providing explicit, conservative evaluation kwargs keeps memory bounded.
        if not skip_evaluation:
            evaluation_kwargs = pipeline_kwargs.get("evaluation_kwargs")
            if evaluation_kwargs is None or not isinstance(evaluation_kwargs, dict):
                evaluation_kwargs = {}
            else:
                evaluation_kwargs = dict(evaluation_kwargs)

            if self.device.type == "cpu":
                # Keep CPU evaluation extremely conservative to avoid OOM for
                # high-dimensional models (e.g., KG-FIT concatenated embeddings).
                evaluation_kwargs.setdefault("batch_size", 1)
                evaluation_kwargs.setdefault("slice_size", 64)
            else:
                evaluation_kwargs.setdefault("batch_size", 64)
                evaluation_kwargs.setdefault("slice_size", 4096)

            pipeline_kwargs = dict(pipeline_kwargs)
            pipeline_kwargs["evaluation_kwargs"] = evaluation_kwargs

        # --------------------------
        # SimKGC backend: separate training path (do not call PyKEEN pipeline)
        # --------------------------
        if embedding_backend == "simkgc":
            if dataset_name is not None:
                raise ValueError("SimKGC backend requires dir_triples (for artifacts preparation).")
            if dir_triples is None:
                raise ValueError("SimKGC backend requires dir_triples.")
            if dir_save is None:
                raise ValueError("SimKGC backend requires dir_save (output directory).")

            simkgc_cfg_payload = pipeline_kwargs.pop("simkgc_config", None)
            if simkgc_cfg_payload is None:
                simkgc_cfg = SimKGCConfig(pretrained_model="__dummy__")
            elif isinstance(simkgc_cfg_payload, dict):
                simkgc_cfg = SimKGCConfig.from_dict(simkgc_cfg_payload)
            else:
                raise ValueError("simkgc_config must be a dict")

            out_dir = Path(dir_save)
            artifacts_dir = out_dir / "simkgc_artifacts"
            out_dir.mkdir(parents=True, exist_ok=True)

            prepare_simkgc_artifacts(
                dir_triples=Path(dir_triples),
                artifacts_dir=artifacts_dir,
            )

            wrapper = SimKGCWrapper(
                config=simkgc_cfg,
                artifacts_dir=artifacts_dir,
                output_dir=out_dir,
            )
            wrapper.train()

            # Return a loaded instance (so score/evaluate works immediately)
            return KnowledgeGraphEmbedding(model_dir=str(out_dir), fit_normalization=False)

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
            if os.path.exists(f_test) and os.path.getsize(f_test) > 0:
                test_tf = TriplesFactory.from_path(path=os.path.join(dir_triples, 'test.txt'), 
                                                    entity_to_id=train_tf.entity_to_id,
                                                    relation_to_id=train_tf.relation_to_id)
            else:
                test_tf = None
            
            f_valid = os.path.join(dir_triples, 'valid.txt')
            if os.path.exists(f_valid) and os.path.getsize(f_valid) > 0:
                valid_tf = TriplesFactory.from_path(path=os.path.join(dir_triples, 'valid.txt'), 
                                                    entity_to_id=train_tf.entity_to_id,
                                                    relation_to_id=train_tf.relation_to_id)
            else:
                valid_tf = None

        # --------------------------
        # guard: early stopper requires validation triples
        # --------------------------
        stopper = pipeline_kwargs.get("stopper")
        if stopper is not None and valid_tf is None:
            logger.warning(
                "Stopper is configured (stopper=%r) but validation triples are missing. "
                "Disabling stopper to avoid training crash.",
                stopper,
            )
            pipeline_kwargs = dict(pipeline_kwargs)
            pipeline_kwargs["stopper"] = None
            pipeline_kwargs.pop("stopper_kwargs", None)

        if skip_evaluation:
            if train_tf is None:
                raise ValueError("skip_evaluation=True requires training triples")
            sample_triples = train_tf.triples[:1]
            test_tf = TriplesFactory.from_labeled_triples(
                sample_triples,
                entity_to_id=train_tf.entity_to_id,
                relation_to_id=train_tf.relation_to_id,
            )
            valid_tf = test_tf
            if stopper is not None:
                logger.warning("skip_evaluation=True: disabling stopper due to tiny validation triples")
                pipeline_kwargs = dict(pipeline_kwargs)
                pipeline_kwargs["stopper"] = None
                pipeline_kwargs.pop("stopper_kwargs", None)

        # --------------------------
        # KG-FIT: load pretrained entity embeddings (full/slice)
        # --------------------------
        if embedding_backend == "kgfit":
            if dataset_name is not None:
                raise ValueError("KG-FIT backend requires dir_triples (precomputed embeddings).")
            if dir_triples is None:
                raise ValueError("KG-FIT backend requires dir_triples.")

            kgfit_config = kgfit_config or {}
            paths = resolve_kgfit_paths(
                dir_triples=Path(dir_triples),
                override=kgfit_config.get("paths") if isinstance(kgfit_config, dict) else None,
            )
            reshape_strategy = kgfit_config.get("reshape_strategy", "full")
            embedding_dim = kgfit_config.get("embedding_dim")

            try:
                pretrained = load_kgfit_entity_embeddings(
                    entity_to_id=train_tf.entity_to_id,
                    config=KGFitEmbeddingConfig(
                        paths=paths,
                        reshape_strategy=reshape_strategy,
                        embedding_dim=embedding_dim,
                    ),
                    dtype=torch.float32,
                )
            except KGFitEmbeddingError as err:
                raise ValueError(f"KG-FIT embeddings unavailable: {err}") from err

            # load seed hierarchy artifacts
            hierarchy_cfg = kgfit_config.get("hierarchy", {})
            hierarchy_path = Path(hierarchy_cfg.get("hierarchy_seed", Path(dir_triples) / ".cache" / "kgfit" / "hierarchy_seed.json"))
            centers_path = Path(hierarchy_cfg.get("cluster_embeddings", Path(dir_triples) / ".cache" / "kgfit" / "cluster_embeddings.npy"))
            neighbors_path = Path(hierarchy_cfg.get("neighbor_clusters", Path(dir_triples) / ".cache" / "kgfit" / "neighbor_clusters.json"))
            artifacts = load_seed_hierarchy_artifacts(
                hierarchy_path=hierarchy_path,
                cluster_centers_path=centers_path,
                neighbor_clusters_path=neighbors_path,
            )

            # align entity -> cluster index
            label_to_cluster_index = {label: idx for idx, label in enumerate(artifacts.cluster_labels)}
            entity_label_map = {entity: label for entity, label in zip(artifacts.entity_ids, artifacts.entity_cluster_labels)}
            entity_cluster_indices = torch.empty(len(train_tf.entity_to_id), dtype=torch.long)
            missing = []
            for entity, entity_id in train_tf.entity_to_id.items():
                label = entity_label_map.get(entity)
                if label is None:
                    missing.append(entity)
                    continue
                entity_cluster_indices[entity_id] = label_to_cluster_index[int(label)]
            if missing:
                raise ValueError(f"Seed hierarchy missing {len(missing)} entities. Example: {missing[:3]}")

            neighbor_k = int(hierarchy_cfg.get("neighbor_k", 5))
            neighbor_clusters = None
            if artifacts.neighbor_clusters and neighbor_k > 0:
                max_k = min(neighbor_k, max(len(v) for v in artifacts.neighbor_clusters.values()))
                neighbor_clusters = torch.zeros(
                    (len(artifacts.cluster_labels), max_k), dtype=torch.long
                )
                for cluster_idx, neighbors in artifacts.neighbor_clusters.items():
                    if neighbors:
                        trimmed = neighbors[:max_k]
                        neighbor_clusters[cluster_idx, : len(trimmed)] = torch.tensor(trimmed, dtype=torch.long)

            regularizer_cfg = kgfit_config.get("regularizer", {})
            kgfit_regularizer = KGFitRegularizer(
                anchor_embeddings=pretrained,
                entity_cluster_indices=entity_cluster_indices,
                cluster_centers=torch.as_tensor(artifacts.cluster_centers, dtype=torch.float32),
                neighbor_clusters=neighbor_clusters,
                config=KGFitRegularizerConfig(
                    anchor_weight=float(regularizer_cfg.get("anchor_weight", 0.5)),
                    cohesion_weight=float(regularizer_cfg.get("cohesion_weight", 0.5)),
                    separation_weight=float(regularizer_cfg.get("separation_weight", 0.5)),
                    separation_margin=float(regularizer_cfg.get("separation_margin", 0.2)),
                    profile=bool(regularizer_cfg.get("profile", False)),
                    profile_every=int(regularizer_cfg.get("profile_every", 200)),
                ),
            )

            # build ERModel with KG-FIT entity representation
            model_kwargs = pipeline_kwargs.get("model_kwargs") or {}
            scoring_norm = int(model_kwargs.get("scoring_fct_norm", 1))
            model_name = model.lower() if isinstance(model, str) else str(model).lower()
            if model_name not in {"transe", "pairre"}:
                raise ValueError("KG-FIT backend currently supports model='transe' or model='pairre' only")

            entity_rep = KGFitEntityEmbedding(
                kgfit_regularizer=kgfit_regularizer,
                max_id=len(train_tf.entity_to_id),
                embedding_dim=int(pretrained.shape[1]),
                initializer=PretrainedInitializer(tensor=pretrained),
                trainable=True,
            )
            embedding_dim = int(pretrained.shape[1])
            if model_name == "transe":
                relation_rep = Embedding(
                    max_id=train_tf.num_relations,
                    embedding_dim=embedding_dim,
                )
                model = ERModel(
                    triples_factory=train_tf,
                    interaction=TransEInteraction(p=scoring_norm),
                    entity_representations=entity_rep,
                    relation_representations=relation_rep,
                )
            else:
                # PairRE uses two relation representation vectors per relation.
                relation_rep_h = Embedding(
                    max_id=train_tf.num_relations,
                    embedding_dim=embedding_dim,
                )
                relation_rep_t = Embedding(
                    max_id=train_tf.num_relations,
                    embedding_dim=embedding_dim,
                )
                model = ERModel(
                    triples_factory=train_tf,
                    interaction=PairREInteraction(p=scoring_norm),
                    entity_representations=entity_rep,
                    relation_representations=[relation_rep_h, relation_rep_t],
                )

            # avoid passing model_kwargs when providing a model instance
            pipeline_kwargs = dict(pipeline_kwargs)
            pipeline_kwargs.pop("model_kwargs", None)

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
        instance.device = self.device

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

    @staticmethod
    def _get_safe_device() -> torch.device:
        """Select CUDA if truly usable; otherwise fall back to CPU.

        torch.cuda.is_available() can be True even when NVML/driver init fails in
        containerized environments. This guard avoids hard failures and hangs.
        """
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible in {"", "-1"}:
            return torch.device("cpu")
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            # Trigger CUDA init / NVML checks early.
            _ = torch.cuda.device_count()
            return torch.device("cuda")
        except Exception as e:
            logger.warning("CUDA appears unavailable at runtime; falling back to CPU. error=%r", e)
            return torch.device("cpu")

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
        skipped = []
        for h, r, t in triples:
            if h not in self.triples.entity_to_id or r not in self.triples.relation_to_id or t not in self.triples.entity_to_id:
                # Skip triples with unknown entities/relations instead of raising error
                skipped.append((h, r, t))
                continue
            h_ids.append(self.triples.entity_to_id[h])
            r_ids.append(self.triples.relation_to_id[r])
            t_ids.append(self.triples.entity_to_id[t])

        if skipped:
            logger.warning(f"Skipped {len(skipped)} triples with unknown entities/relations")
        
        if not h_ids:
            # All triples were skipped
            return torch.tensor([], dtype=torch.long, device=self.device).reshape(0, 3)
        
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
        if self.embedding_backend == "simkgc":
            if self._simkgc is None:
                raise RuntimeError("SimKGC backend not initialized")
            arr = self._simkgc.score_triples(
                triples=labeled_triples,
                entity_text=self._simkgc_entity_text,
                relation_text=self._simkgc_relation_text,
                normalize_0_1=bool(normalize),
            )
            return [float(x) for x in arr.tolist()]

        triples = self._label_to_id(labeled_triples)
        
        # Handle empty triples (all skipped)
        if len(triples) == 0:
            logger.warning("No valid triples to score (all were skipped)")
            return []

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
    
    def _fit_normalization(self, batch_size: int = 4096) -> None:
        """Fit normalization parameters based on training triples scores.

        Args:
            batch_size: Batch size for scoring to avoid large allocations.
        """
        scores_min = float("inf")
        scores_max = float("-inf")
        mapped = self.triples.mapped_triples.to(self.device)
        with torch.no_grad():
            for batch in mapped.split(split_size=batch_size):
                scores = self.model.score_hrt(batch)
                batch_min = float(scores.min().item())
                batch_max = float(scores.max().item())
                scores_min = min(scores_min, batch_min)
                scores_max = max(scores_max, batch_max)
        self._train_score_min = scores_min
        self._train_score_max = scores_max
    
    def evaluate(
        self,
        test_triples: Optional[TriplesFactory] = None,
        filtered: bool = True,
        batch_size: int = 256,
        slice_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on test triples using rank-based metrics (Hits@k, MRR).
        
        Args:
            test_triples: TriplesFactory containing test triples. If None, uses the model's own test set if available.
            batch_size: Batch size for evaluation.
            slice_size: Slice size for scoring entities to limit memory usage.
            
        Returns:
            Dictionary containing evaluation metrics:
                - hits_at_1: Hits@1
                - hits_at_3: Hits@3
                - hits_at_10: Hits@10
                - mean_reciprocal_rank: MRR
        """
        if self.embedding_backend == "simkgc":
            if self._simkgc is None:
                raise RuntimeError("SimKGC backend not initialized")
            metrics = self._simkgc.evaluate(split="test")
            return {
                "hits_at_1": float(metrics.hits_at_1),
                "hits_at_3": float(metrics.hits_at_3),
                "hits_at_10": float(metrics.hits_at_10),
                "mean_reciprocal_rank": float(metrics.mrr),
            }

        # If no test triples provided, try to load from the model directory
        if test_triples is None:
            # Check if test.txt exists in the model directory
            if hasattr(self, 'dir_save') and self.dir_save:
                test_path = os.path.join(self.dir_save, 'test.txt')
                if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                    test_triples = TriplesFactory.from_path(
                        path=test_path,
                        entity_to_id=self.triples.entity_to_id,
                        relation_to_id=self.triples.relation_to_id
                    )
                else:
                    logger.warning("No test triples provided and test.txt not found. Using validation set if available.")
                    # Try validation set
                    valid_path = os.path.join(self.dir_save, 'valid.txt')
                    if os.path.exists(valid_path) and os.path.getsize(valid_path) > 0:
                        test_triples = TriplesFactory.from_path(
                            path=valid_path,
                            entity_to_id=self.triples.entity_to_id,
                            relation_to_id=self.triples.relation_to_id
                        )
                    else:
                        raise ValueError("No test or validation triples available for evaluation")
            else:
                raise ValueError("Model directory not set and no test triples provided")
        
        # Create evaluator
        # NOTE: PyKEEN uses torch_max_mem for automatic memory optimization.
        # In CPU-only environments, this can attempt aggressive allocations and
        # trigger the OOM killer. We disable it and rely on explicit batch/slice
        # sizes for stability.
        evaluator = RankBasedEvaluator(
            filtered=bool(filtered),
            automatic_memory_optimization=False,
        )

        def _is_cuda_oom(error: BaseException) -> bool:
            message = str(error)
            return (
                "out of memory" in message.lower()
                or "cuda error" in message.lower() and "memory" in message.lower()
            )

        def _evaluate_on_device(
            device: torch.device,
            eval_batch_size: int,
            eval_slice_size: Optional[int],
        ):
            mapped_triples = test_triples.mapped_triples.to(device)
            all_pos_triples = None
            if filtered:
                # Avoid calling Evaluator.evaluate() / evaluator.evaluate(), since
                # PyKEEN internally starts with batch_size=num_triples and relies
                # on torch_max_mem to reduce it on OOM. On CPU, OOM can trigger
                # the OS OOM killer (exit 137) before any exception is raised.
                from pykeen.utils import prepare_filter_triples

                additional_filter_triples = self.triples.mapped_triples.to(device)
                all_pos_triples = prepare_filter_triples(
                    mapped_triples=mapped_triples,
                    additional_filter_triples=additional_filter_triples,
                ).to(device)
            self.model = self.model.to(device)

            if device.type == "cpu":
                # Reduce thread-level memory spikes on CPU.
                try:
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass

            # Safe evaluation loop (no torch_max_mem batch/slice probing)
            from pykeen.constants import LABEL_HEAD, LABEL_TAIL
            from pykeen.evaluation.evaluator import _evaluate_batch

            evaluator.clear()
            targets = (LABEL_HEAD, LABEL_TAIL)
            with torch.inference_mode():
                for batch in mapped_triples.split(split_size=eval_batch_size):
                    relation_filter = None
                    for target in targets:
                        relation_filter = _evaluate_batch(
                            batch=batch,
                            target=target,
                            evaluator=evaluator,
                            relation_filter=relation_filter,
                            # kwargs expected by _evaluate_batch
                            model=self.model,
                            all_pos_triples=all_pos_triples,
                            restrict_entities_to=None,
                            slice_size=eval_slice_size,
                            mode=evaluator.mode,
                        )
            return evaluator.finalize()

        # Evaluate (with CUDA OOM fallback)
        logger.info("Evaluating model on test set...")
        try:
            device = torch.device(self.device)
        except Exception:
            device = torch.device("cpu")

        # Default to conservative settings on CPU to avoid large allocations.
        if device.type == "cpu":
            if slice_size is None:
                slice_size = 64
            batch_size = min(int(batch_size), 2)

        try:
            results = _evaluate_on_device(device, batch_size, slice_size)
        except Exception as e:
            if device.type == "cuda" and _is_cuda_oom(e):
                logger.warning(
                    "CUDA OOM during evaluation. Retrying with smaller batch/slice sizes, then CPU fallback if needed. (%s)",
                    str(e).splitlines()[0] if str(e) else repr(e),
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                retry_settings = [
                    (64, slice_size if slice_size is not None else 4096),
                    (16, 2048),
                ]
                results = None
                last_error: Optional[BaseException] = None

                for bs, ss in retry_settings:
                    try:
                        results = _evaluate_on_device(device, bs, ss)
                        break
                    except Exception as e2:
                        last_error = e2
                        if _is_cuda_oom(e2):
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            continue
                        raise

                if results is None:
                    logger.warning(
                        "GPU evaluation still failing (likely OOM). Falling back to CPU evaluation. (%s)",
                        str(last_error).splitlines()[0] if last_error and str(last_error) else repr(last_error),
                    )
                    cpu_device = torch.device("cpu")
                    results = _evaluate_on_device(cpu_device, min(batch_size, 128), slice_size)
            else:
                raise
        
        # Extract metrics
        metrics = {
            'hits_at_1': results.get_metric('hits@1'),
            'hits_at_3': results.get_metric('hits@3'),
            'hits_at_10': results.get_metric('hits@10'),
            'mean_reciprocal_rank': results.get_metric('mean_reciprocal_rank')
        }
        
        logger.info(f"Evaluation results: Hits@1={metrics['hits_at_1']:.4f}, "
                   f"Hits@3={metrics['hits_at_3']:.4f}, "
                   f"Hits@10={metrics['hits_at_10']:.4f}, "
                   f"MRR={metrics['mean_reciprocal_rank']:.4f}")
        
        return metrics
