"""Concrete rule extractor implementations."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from simple_active_refine.amie import AmieRule, AmieRules, LLMRuleFilter
from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.pipeline import BaseRuleExtractor, RuleExtractionContext, RuleExtractionResult
from simple_active_refine.rule_extractor import extract_rules_from_entire_graph, extract_rules_from_high_score_triples
from simple_active_refine.util import get_logger

logger = get_logger(__name__)


def select_diverse_rules(amie_rules, n_rules: int) -> List[AmieRule]:
    """Select rules using PCA, head coverage, and simplicity composite score."""
    if len(amie_rules.rules) == 0:
        return []

    pca_scores = np.array([r.pca_conf if r.pca_conf is not None else 0.0 for r in amie_rules.rules])
    coverage_scores = np.array([r.head_coverage if r.head_coverage is not None else 0.0 for r in amie_rules.rules])
    body_sizes = np.array([r.body_size if r.body_size is not None else 1 for r in amie_rules.rules])

    pca_norm = (pca_scores - pca_scores.min()) / (pca_scores.max() - pca_scores.min() + 1e-10)
    coverage_norm = (coverage_scores - coverage_scores.min()) / (coverage_scores.max() - coverage_scores.min() + 1e-10)
    diversity_norm = 1.0 / (body_sizes + 1.0)
    diversity_norm = (diversity_norm - diversity_norm.min()) / (diversity_norm.max() - diversity_norm.min() + 1e-10)

    composite = 0.4 * pca_norm + 0.3 * coverage_norm + 0.3 * diversity_norm
    top_indices = np.argsort(composite)[::-1][:n_rules]
    return [amie_rules.rules[i] for i in top_indices]


class HighScoreRuleExtractor(BaseRuleExtractor):
    """Rule extractor that trains a KGE per iteration and mines AMIE+ rules."""

    def __init__(
        self,
        data_manager,
        embedding_config: Dict,
        target_relation: str,
        use_high_score_triples: bool,
        lower_percentile: float,
        k_neighbor: int,
        min_head_coverage: float,
        min_pca_conf: float,
        n_rules_pool: int,
        tmp_dir: str,
        use_llm_filter: bool = False,
        llm_model: str = "gpt-4o",
        llm_temperature: float = 0.0,
        llm_request_timeout: int = 120,
        llm_max_tokens: int = 16000,
        llm_top_k: Optional[int] = None,
        llm_min_pca_conf: Optional[float] = None,
        llm_min_head_coverage: Optional[float] = None,
    ) -> None:
        self.data_manager = data_manager
        self.embedding_config = embedding_config
        self.target_relation = target_relation
        self.use_high_score_triples = use_high_score_triples
        self.lower_percentile = lower_percentile
        self.k_neighbor = k_neighbor
        self.min_head_coverage = min_head_coverage
        self.min_pca_conf = min_pca_conf
        self.n_rules_pool = n_rules_pool
        self.tmp_dir = tmp_dir
        self.use_llm_filter = use_llm_filter
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_request_timeout = llm_request_timeout
        self.llm_max_tokens = llm_max_tokens
        self.llm_top_k = llm_top_k
        self.llm_min_pca_conf = llm_min_pca_conf
        self.llm_min_head_coverage = llm_min_head_coverage

    def extract(self, context: RuleExtractionContext) -> RuleExtractionResult:
        iteration_idx = max(context.iteration - 1, 0)
        iter_dir = self.data_manager.write_iteration(iteration_idx, context.kg)

        logger.info("[v3] Training KGE for rule extraction")
        config = deepcopy(self.embedding_config)
        config["dir_triples"] = iter_dir
        config["dir_save"] = iter_dir
        kge = KnowledgeGraphEmbedding.train_model(**config)

        if self.use_high_score_triples:
            logger.info(
                "[v3] Using high-score triples for AMIE+ (percentile=%.1f, k=%d)",
                self.lower_percentile,
                self.k_neighbor,
            )
            amie_rules = extract_rules_from_high_score_triples(
                kge=kge,
                target_relation=self.target_relation,
                lower_percentile=self.lower_percentile,
                k_neighbor=self.k_neighbor,
                min_head_coverage=self.min_head_coverage,
                min_pca_conf=self.min_pca_conf,
                dir_working=self.tmp_dir,
            )
        else:
            logger.info("[v3] Using entire graph for AMIE+")
            amie_rules = extract_rules_from_entire_graph(
                kge=kge,
                target_relation=self.target_relation,
                dir_triples=iter_dir,
                min_head_coverage=self.min_head_coverage,
                min_pca_conf=self.min_pca_conf,
            )

        n_rules_before_llm = len(amie_rules.rules)
        if self.use_llm_filter:
            logger.info(
                "[v3] Filtering rules with LLMRuleFilter (model=%s, top_k=%s)",
                self.llm_model,
                self.llm_top_k or self.n_rules_pool,
            )
            llm_filter = LLMRuleFilter(
                model=self.llm_model,
                temperature=self.llm_temperature,
                request_timeout=self.llm_request_timeout,
                max_tokens=self.llm_max_tokens,
            )
            filtered_rules = llm_filter.filter(
                rules=amie_rules.rules,
                min_pca_conf=self.llm_min_pca_conf,
                min_head_coverage=self.llm_min_head_coverage,
                top_k=self.llm_top_k or self.n_rules_pool,
            )
            amie_rules = filtered_rules if isinstance(filtered_rules, AmieRules) else AmieRules(filtered_rules)
            selected = amie_rules.rules
        else:
            selected = select_diverse_rules(amie_rules, self.n_rules_pool)
        diagnostics = {
            "n_rules_before_llm": n_rules_before_llm,
            "n_rules_extracted": len(amie_rules.rules),
            "n_rules_selected": len(selected),
            "use_llm_filter": self.use_llm_filter,
        }
        dump_path = os.path.join(iter_dir, "rule_extractor_io.json")
        try:
            payload = {
                "iteration": context.iteration,
                "input": {
                    "use_high_score_triples": self.use_high_score_triples,
                    "lower_percentile": self.lower_percentile,
                    "k_neighbor": self.k_neighbor,
                    "min_head_coverage": self.min_head_coverage,
                    "min_pca_conf": self.min_pca_conf,
                    "n_rules_pool": self.n_rules_pool,
                    "target_relation": self.target_relation,
                },
                "output": {
                    "selected_rules": [str(r) for r in selected],
                    "all_rules_count": len(amie_rules.rules),
                    "selected_count": len(selected),
                    "diagnostics": diagnostics,
                },
            }
            with open(dump_path, "w", encoding="utf-8") as fout:
                json.dump(payload, fout, ensure_ascii=False, indent=2)
        except Exception as err:
            logger.warning("[v3] Failed to dump rule extractor IO: %s", err)
        return RuleExtractionResult(rules=selected, diagnostics=diagnostics)


class PrecomputedRuleExtractor(BaseRuleExtractor):
    """Return a fixed, precomputed rule pool on every extraction call."""

    def __init__(self, rules: List[AmieRule], diagnostics: Optional[Dict] = None) -> None:
        self.rules = rules
        self.diagnostics = diagnostics or {}

    def extract(self, context: RuleExtractionContext) -> RuleExtractionResult:
        diagnostics = dict(self.diagnostics)
        diagnostics.update({
            "precomputed": True,
            "n_rules": len(self.rules),
            "iteration": context.iteration,
        })
        logger.info("[v3] Using precomputed rule pool (size=%d)", len(self.rules))
        return RuleExtractionResult(rules=self.rules, diagnostics=diagnostics)
