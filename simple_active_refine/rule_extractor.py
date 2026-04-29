# TODO 最終的にはrule generatorとマージすること
from __future__ import annotations
"""
Extract "triple addition" rules R_add via AMIE+
----------------------------------------------
Usage:
  python extract_rules.py DIR_EMBEDDING TARGET_RELATION SCORE_THRESHOLD DIR_RULES \
         --dir-triples DIR_TRIPLES --amie-jar /path/amie.jar --k 2

Note: DIR_EMBEDDING is used here only to locate the original triples via
--dir-triples; rule mining uses the merged k‑hop subgraph from high‑score
examples (threshold on model score needs the model; if unavailable, you may
supply a fixed list or skip scoring and use top‑degree examples).
"""


import shutil
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import logging
import uuid
import json
import os
import concurrent.futures
import multiprocessing

from settings import PATH_AMIE_JAR
from simple_active_refine.io_utils import write_triples
from simple_active_refine.subgraph import extract_k_hop_enclosing_subgraph
from simple_active_refine.amie import AmieRules
from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.util import get_logger

# ログの基本設定
logger = get_logger('rule extractor')

def get_related_triples(target_triple: Tuple[str, str, str], 
                        triples: List[Tuple[str, str, str]], 
                        k: int,
                        remove_target: bool = True,
                        directed=False) -> Tuple[List[Tuple[str, str, str]], Tuple[str, str, str]]:
    """
    Extract k-hop enclosing subgraph around the target triple.
    Assign new UUIDs to entities to avoid ID collisions when merging subgraphs.
    Args:
        target_triple: The target triple (h, r, t).
        triples: List of all triples in the KG.
        k: Number of hops for the enclosing subgraph.
        remove_target: Whether to remove the target triple from the subgraph.
        directed: Whether to treat the graph as directed.
    Returns:
        A tuple containing:
        - List of triples in the k-hop subgraph with new UUIDs.
        - The original target triple.
    """
    sub = extract_k_hop_enclosing_subgraph(triples, 
                                           target_triple, 
                                           k=k, directed=directed, remove_target=remove_target)
    uuid_prefix = str(uuid.uuid4()) + "_"
    triples_out = [(uuid_prefix + str(h), r, uuid_prefix + str(t)) for h, r, t in sub]
    return triples_out, target_triple


def extract_rules_from_entire_graph(kge: KnowledgeGraphEmbedding,
                                     target_relation: str,
                                     top_k: int = 3,
                                     sorted_by: str = None,
                                     min_head_coverage: float = 0.01,
                                     min_pca_conf: float = 0.01,
                                     dir_working: str = './tmp') -> AmieRules:
    """
    Extract AMIE+ rules from the entire knowledge graph.
    
    Args:
        kge: Knowledge graph embedding model
        target_relation: Target relation to extract rules for
        top_k: Number of top rules to return
        sorted_by: Sort metric for filtering rules
        min_head_coverage: Minimum head coverage threshold for AMIE+
        min_pca_conf: Minimum PCA confidence threshold for AMIE+
        dir_working: Working directory for temporary files
        
    Returns:
        AmieRules: Extracted rules filtered by target relation
    """
    logger.info("Extracting AMIE+ rules from entire knowledge graph...")
    
    # Get all triples from the knowledge graph
    all_triples = kge.get_labeled_triples()
    logger.info(f"Total triples in knowledge graph: {len(all_triples)}")
    
    # Count target relation triples
    target_triples = [triple for triple in all_triples if triple[1] == target_relation]
    logger.info(f"Target relation '{target_relation}' triples: {len(target_triples)}")
    
    # Create temporary directory for AMIE+
    dir_rules = os.path.join(dir_working, f"rules_entire_{str(uuid.uuid4())[:8]}")
    logger.info(f'Create temporary directory for AMIE+: {dir_rules}')
    os.makedirs(dir_rules, exist_ok=True)
    
    # Save all triples for AMIE+
    amie_in = os.path.join(dir_rules, "entire_graph.tsv")
    logger.info(f"Writing entire graph triples to {amie_in}")
    write_triples(amie_in, all_triples)
    
    # Run AMIE+ on entire graph
    logger.info("Running AMIE+ on entire knowledge graph...")
    rules = AmieRules.run_amie(
        all_triples,
        amie_jar=PATH_AMIE_JAR,
        min_head_coverage=min_head_coverage,
        min_pca=min_pca_conf,
        java_opts=["-Xmx8G"],  # Increased memory for larger graph
    )
    
    # Save all rules before filtering
    rules.to_csv(os.path.join(dir_rules, 'amie_rules_all.csv'))
    logger.info(f"Total rules extracted: {len(rules.rules)}")
    
    # Filter by target relation
    rules = rules.filter_rules_by_head_relation(target_relation)
    logger.info(f"Rules with head relation '{target_relation}': {len(rules.rules)}")
    
    # Exclude gardening_hint relations (semantically meaningless)
    rules = rules.exclude_relations_by_pattern(['/dataworld/gardening_hint/'])
    logger.info(f"Rules after excluding gardening_hint relations: {len(rules.rules)}")
    
    # Save filtered rules
    rules.to_csv(os.path.join(dir_rules, 'amie_rules_filtered.csv'))
    
    # Apply additional filtering if specified
    if sorted_by and len(rules.rules) > 0:
        logger.info(f"Filtering rules by {sorted_by}, keeping top {top_k}...")
        rules = rules.filter(sort_by=sorted_by, top_k=top_k)
    
    return rules


def extract_rules_from_high_score_triples(kge:KnowledgeGraphEmbedding, 
                                          target_relation:str, 
                                          top_k=3, 
                                          lower_percentile=90,
                                          k_neighbor=1,
                                          sorted_by=None,
                                          min_head_coverage=0.01,
                                          min_pca_conf = 0.01,
                                          dir_working='./tmp'):
        
    # --------------------------------------------
    # 準備
    # --------------------------------------------
    # 知識グラフ埋込モデルの読み込み
    all_triples = kge.get_labeled_triples()
    target_triples = [triple for triple in all_triples if triple[1] == target_relation]

    # スコア分布の確認
    all_scores = kge.score_triples(target_triples)
    logger.info(f"Total target triples: {len(target_triples)}")
    logger.info(f"Score stats: min {min(all_scores)}, max {max(all_scores)}, mean {sum(all_scores)/len(all_scores)}")

    # --------------------------------------------
    # スコアが高い上位n%のトリプルを選択
    # --------------------------------------------
    list_triples_and_score \
        = kge.filter_triples_by_score(labeled_triples=target_triples, 
                                      lower_percentile=lower_percentile, 
                                      return_with_score=True)
    target_triples_with_high_score = [triple for triple, _ in list_triples_and_score]
    logger.info(f"Number of triples above {lower_percentile}th percentile: {len(list_triples_and_score)}")
    logger.info(f"Score range of selected triples: {min(score for _, score in list_triples_and_score)} - {max(score for _, score in list_triples_and_score)}")  


    # --------------------------------------------
    # k-hop囲い込みグラフの抽出
    # --------------------------------------------
    # k-hop囲い込みグラフを並列で抽出
    dict_triple_and_k_hop_subgraph = dict()
    # NOTE: Passing a large `all_triples` list to many worker processes can be very expensive
    # due to pickling. When only a few target triples are selected, run sequentially.
    max_workers_default = min(30, multiprocessing.cpu_count())
    n_tasks = len(target_triples_with_high_score)
    num_workers = min(max_workers_default, max(1, n_tasks))
    logger.info(
        f"Extracting {k_neighbor}-hop enclosing subgraphs around {n_tasks} target triples using {num_workers} workers..."
    )

    if n_tasks == 0:
        logger.warning("No high-score target triples selected; AMIE+ input will be empty.")
    elif num_workers <= 1:
        for target_triple in tqdm(target_triples_with_high_score, total=n_tasks):
            triples, _ = get_related_triples(target_triple, all_triples, k_neighbor)
            key = "\t".join(map(str, target_triple))
            dict_triple_and_k_hop_subgraph[key] = [list(triple) for triple in triples]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(get_related_triples, target_triple, all_triples, k_neighbor)
                for target_triple in target_triples_with_high_score
            ]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                triples, target_triple = f.result()
                key = "\t".join(map(str, target_triple))
                dict_triple_and_k_hop_subgraph[key] = [list(triple) for triple in triples]

    # jsonで保存
    dir_rules = os.path.join(dir_working, f"rules_{str(uuid.uuid4())[:8]}")
    logger.info(f'Create temporary directory for amie: {dir_rules}.')
    if not os.path.exists(dir_rules):
        os.makedirs(dir_rules)
    with open(os.path.join(dir_rules, "k_hop_subgraph.json"), "w") as f:
        json.dump(dict_triple_and_k_hop_subgraph, f, indent=2)
    logger.info(f"Saved related triples to {dir_rules}/k_hop_subgraph.json")


    # --------------------------------------------
    # AMNIEによるルールを抽出
    # --------------------------------------------
    # ルール抽出用にk-hop囲い込みグラフをマージして保存
    amie_in = os.path.join(dir_rules, "amie_subgraph.tsv")
    logger.info("Writing merged subgraph triples to %s", amie_in)
    related_triples = []
    for triples in dict_triple_and_k_hop_subgraph.values():
        related_triples.extend([tuple(triple) for triple in triples])
    logger.info(f"Total triples in merged subgraph: {len(related_triples)}")
    write_triples(amie_in, related_triples)

    # amie+でルール抽出
    logger.info("Running AMIE+ to extract rules using mine_rules_with_amie...")

    rules = AmieRules.run_amie(
        related_triples,
        amie_jar=PATH_AMIE_JAR,
        min_head_coverage=min_head_coverage,
        min_pca=min_pca_conf,
        java_opts=["-Xmx4G"],
    )
    rules.to_csv(os.path.join(dir_rules, 'amie_rules_before_filter.csv'))
    rules = rules.filter_rules_by_head_relation(target_relation)
    
    # Exclude gardening_hint relations (semantically meaningless)
    rules = rules.exclude_relations_by_pattern(['/dataworld/gardening_hint/'])
    logger.info(f"Rules after excluding gardening_hint relations: {len(rules.rules)}")

    # dir_rulesを削除
    #if os.path.exists(dir_rules):
    #    shutil.rmtree(dir_rules)
    #    logger.info(f"Deleted directory: {dir_rules}")
    

    if sorted_by:
        # ルールをフィルタリング
        logger.info("Filtering extracted rules...")
        rules = rules.filter(sort_by = sorted_by ,
                            top_k = top_k)

    return rules