"""
メインスクリプト - 改善版 (v2)

改善内容:
1. 高品質ルール抽出: 
   - スコアの高いトリプルのk-hop近傍から抽出
   - 実際に有用なパターンに基づくルール
2. 複合的なルール選択基準:
   - PCA confidence、head coverage、body sizeを考慮
   - バランスの取れたルールプール構築
3. Early Stopping:
   - スコアが悪化し続ける場合は停止
   - 無駄な計算を削減
4. ルールの多様性確保:
   - 異なる評価軸でルールを選択
   - より広範なパターンをカバー

注意: 候補トリプルに新規エンティティが含まれるため、
      埋込モデルでのスコアリングは不可能。
      高品質なルールから抽出されたトリプルをそのまま使用。

多腕バンディット戦略を用いたルール選択による知識グラフ改善。
"""

import argparse
import json
import os
import random
import shutil
from typing import List, Tuple

from simple_active_refine.rule_generator import BaseRuleGenerator
from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.triples_editor import add_triples_for_single_rule
from simple_active_refine.analyzer import RuleWiseAnalyzer
from simple_active_refine.rule_extractor import extract_rules_from_high_score_triples, extract_rules_from_entire_graph
from simple_active_refine.rule_history import RuleHistory, RuleEvaluationRecord
from simple_active_refine.rule_selector import create_rule_selector, RuleWithId
from simple_active_refine.evaluation import IterationEvaluator
from simple_active_refine.util import get_logger
from pykeen.triples import TriplesFactory

logger = get_logger('main_v2')


def save_markdown(md_text: str, file_path: str):
    """Markdownテキストをファイルに保存
    
    Args:
        md_text: Markdownテキスト
        file_path: 保存先ファイルパス
    """
    with open(file_path, 'w') as fout:
        fout.write(md_text)


def sample_target_triples(all_target_triples, n_sample, exclude_triples=None):
    """target tripleをランダムにサンプリング
    
    Args:
        all_target_triples: 全target tripleのリスト
        n_sample: サンプリング数
        exclude_triples: 除外するトリプルのセット
        
    Returns:
        List: サンプリングされたtarget tripleのリスト
    """
    if exclude_triples is None:
        exclude_triples = set()
    
    available = [t for t in all_target_triples if t not in exclude_triples]
    n_sample = min(n_sample, len(available))
    
    return random.sample(available, n_sample)


def select_diverse_rules(
    amie_rules,
    n_rules: int,
    pca_weight: float = 0.4,
    coverage_weight: float = 0.3,
    diversity_weight: float = 0.3
):
    """
    【改善1】複合的な基準でルールを選択
    
    PCA confidence、head coverage、body diversityを考慮して
    バランスの取れたルールプールを構築する。
    
    Args:
        amie_rules: AMIE+抽出ルール
        n_rules: 選択するルール数
        pca_weight: PCA confidenceの重み
        coverage_weight: Head coverageの重み
        diversity_weight: Body diversityの重み（body sizeの逆数）
        
    Returns:
        選択されたルールのリスト
    """
    import numpy as np
    
    if len(amie_rules.rules) == 0:
        return []
    
    logger.info(f'  [DIVERSE SELECT] Selecting {n_rules} rules from {len(amie_rules.rules)} candidates')
    
    # 各指標を正規化
    pca_scores = np.array([r.pca_conf if r.pca_conf is not None else 0.0 for r in amie_rules.rules])
    coverage_scores = np.array([r.head_coverage if r.head_coverage is not None else 0.0 for r in amie_rules.rules])
    body_sizes = np.array([r.body_size if r.body_size is not None else 1 for r in amie_rules.rules])
    
    # 正規化（0-1範囲）
    pca_norm = (pca_scores - pca_scores.min()) / (pca_scores.max() - pca_scores.min() + 1e-10)
    coverage_norm = (coverage_scores - coverage_scores.min()) / (coverage_scores.max() - coverage_scores.min() + 1e-10)
    # Body sizeは小さいほど良い（シンプルなルール）
    diversity_norm = 1.0 / (body_sizes + 1.0)
    diversity_norm = (diversity_norm - diversity_norm.min()) / (diversity_norm.max() - diversity_norm.min() + 1e-10)
    
    # 複合スコア計算
    composite_scores = (
        pca_weight * pca_norm +
        coverage_weight * coverage_norm +
        diversity_weight * diversity_norm
    )
    
    # 上位n_rules個を選択
    top_indices = np.argsort(composite_scores)[::-1][:n_rules]
    selected_rules = [amie_rules.rules[i] for i in top_indices]
    
    logger.info(f'  [DIVERSE SELECT] Selected {len(selected_rules)} rules')
    for i, idx in enumerate(top_indices[:min(5, n_rules)]):
        rule = amie_rules.rules[idx]
        logger.info(f'    Top {i+1}: PCA={rule.pca_conf:.3f}, Coverage={rule.head_coverage:.3f}, '
                   f'BodySize={rule.body_size}, Score={composite_scores[idx]:.3f}')
    
    return [RuleWithId(rule_id=f'rule_{i:03d}', rule=rule) 
            for i, rule in enumerate(selected_rules)]


def check_iteration_quality(
    target_score_history: List[float],
    patience: int = 2
) -> bool:
    """
    【改善2】Early Stopping判定
    
    対象トリプルのスコアが連続して悪化している場合、改善の見込みがないと判断。
    
    Args:
        target_score_history: 各iterationの対象トリプル平均スコア履歴
        patience: 許容する連続悪化回数
        
    Returns:
        True: 継続すべき、False: 停止すべき
    """
    if len(target_score_history) < patience + 1:
        return True  # まだ判定できない
    
    # 直近patience回が全て悪化しているかチェック
    recent = target_score_history[-(patience+1):]
    is_degrading = all(recent[i] < recent[i-1] for i in range(1, len(recent)))
    
    if is_degrading:
        logger.warning(f'  [EARLY STOP] Target score has degraded for {patience} consecutive iterations')
        logger.warning(f'  [EARLY STOP] Score history: {recent}')
        return False
    
    return True


if __name__ == '__main__':
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Knowledge Graph Improvement with Multi-Armed Bandit (Improved v2)')
    parser.add_argument('--n_iter', type=int, default=3, help='Number of iterations')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for embedding training')
    parser.add_argument('--dir', type=str, default='./experiments/20251214/improved_v2', help='Working directory')
    parser.add_argument('--use_high_score_triples', action='store_true', help='Use high-score triples for rule extraction')
    parser.add_argument('--lower_percentile', type=float, default=80.0, help='Lower percentile for high-score triple selection')
    parser.add_argument('--k_neighbor', type=int, default=1, help='K-hop neighborhood for subgraph extraction')
    parser.add_argument('--early_stop_patience', type=int, default=2, help='Patience for early stopping')
    args = parser.parse_args()
    
    """パラメータ設定"""
    # 対象知識グラフとリレーション
    knowledge_graph = 'FB15k-237'
    target_relation = '/people/person/nationality'
    
    # 全体制御
    dir_initial_triples = './experiments/test_data_for_nationality_v3'
    n_iter = args.n_iter
    dir_working = args.dir
    
    # 【改善パラメータ】
    use_high_score_triples = args.use_high_score_triples  # 高スコアトリプルからルール抽出
    lower_percentile = args.lower_percentile              # 高スコアトリプルの閾値
    k_neighbor = args.k_neighbor                          # k-hop近傍
    early_stop_patience = args.early_stop_patience        # Early Stopping許容回数
    
    # ルールpool設定
    n_rules_pool = 12
    n_rules_select = 3
    n_targets_per_rule = 10
    
    # ルール選択戦略
    rule_selector_strategy = 'llm_policy'
    llm_temperature = 0.3
    
    # AMIE+設定
    use_amie_rules = True
    min_head_coverage = 0.01
    min_pca_conf = 0.05  # 初期フィルタは緩めに設定（多様性確保）
    
    # 埋込モデル設定
    f_config_embedding = "./config_embeddings.json"
    num_epochs = args.num_epochs
    
    # 作業ディレクトリ作成
    if not os.path.exists(dir_working):
        os.makedirs(dir_working, exist_ok=True)
    
    # 設定ファイル読み込み
    with open(f_config_embedding) as fin:
        config_embedding = json.load(fin)
    config_embedding["training_kwargs"]["num_epochs"] = num_epochs
    
    with open(os.path.join(dir_initial_triples, 'config_dataset.json'), 'r') as fin:
        config_dataset = json.load(fin)
    
    logger.info('='*80)
    logger.info('Phase 0: Initialization (v2 - Improved)')
    logger.info('='*80)
    logger.info(f'[IMPROVEMENT] High-score triple based rule extraction: {use_high_score_triples}')
    if use_high_score_triples:
        logger.info(f'[IMPROVEMENT]   - Lower percentile: {lower_percentile}')
        logger.info(f'[IMPROVEMENT]   - K-hop neighbor: {k_neighbor}')
    logger.info(f'[IMPROVEMENT] Diverse rule selection enabled (PCA + Coverage + Simplicity)')
    logger.info(f'[IMPROVEMENT] Early stopping enabled (patience={early_stop_patience})')
    
    # Iteration 0の作成
    dir_iter_0 = os.path.join(dir_working, 'iter_0')
    if os.path.exists(dir_iter_0):
        logger.info(f'Removing existing directory {dir_iter_0}')
        shutil.rmtree(dir_iter_0)
    shutil.copytree(dir_initial_triples, dir_iter_0)
    
    # 初期埋込モデル学習
    logger.info('Step 0.1: Training initial knowledge graph embedding')
    logger.info(f'  [INPUT] Training data directory: {dir_iter_0}')
    logger.info(f'  [INPUT] Model: {config_embedding["model"]}')
    logger.info(f'  [INPUT] Embedding dim: {config_embedding.get("model_kwargs", {}).get("embedding_dim", "N/A")}')
    logger.info(f'  [INPUT] Epochs: {num_epochs}')
    config_embedding['dir_triples'] = dir_iter_0
    config_embedding['dir_save'] = dir_iter_0
    kge_initial = KnowledgeGraphEmbedding.train_model(**config_embedding)
    logger.info(f'  [OUTPUT] Model saved to: {dir_iter_0}')
    logger.info(f'  [OUTPUT] Model type: {type(kge_initial).__name__}')
    
    # AMIE+ルール抽出
    logger.info('Step 0.2: Extracting AMIE+ rules for initial rule pool')
    if use_amie_rules:
        logger.info(f'  [INPUT] Target relation: {config_dataset["target_relation"]}')
        logger.info(f'  [INPUT] Min PCA confidence: {min_pca_conf}')
        logger.info(f'  [INPUT] Min head coverage: {min_head_coverage}')
        
        if use_high_score_triples:
            # 【改善】高スコアトリプルのk-hop近傍からルール抽出
            logger.info(f'  [METHOD] Using high-score triples (percentile={lower_percentile}, k={k_neighbor})')
            amie_rules = extract_rules_from_high_score_triples(
                kge=kge_initial,
                target_relation=config_dataset['target_relation'],
                lower_percentile=lower_percentile,
                k_neighbor=k_neighbor,
                min_head_coverage=min_head_coverage,
                min_pca_conf=min_pca_conf,
                dir_working='./tmp'
            )
        else:
            # オリジナル：知識グラフ全体からルール抽出
            logger.info(f'  [METHOD] Using entire graph')
            amie_rules = extract_rules_from_entire_graph(
                kge=kge_initial,
                target_relation=config_dataset['target_relation'],
                dir_triples=dir_iter_0,
                min_head_coverage=min_head_coverage,
                min_pca_conf=min_pca_conf
            )
        
        logger.info(f'  [OUTPUT] Extracted {len(amie_rules.rules)} AMIE+ rules')
        
        # 【改善】複合的な基準でルール選択
        logger.info(f'  [SELECTION] Using diverse selection criteria')
        rule_pool = select_diverse_rules(
            amie_rules=amie_rules,
            n_rules=n_rules_pool,
            pca_weight=0.4,
            coverage_weight=0.3,
            diversity_weight=0.3
        )
        
        logger.info(f'  [OUTPUT] Created rule pool with {len(rule_pool)} rules')
    else:
        logger.info('  [SKIP] AMIE+ disabled, will generate rules with LLM')
        rule_generator = BaseRuleGenerator(knowledge_graph=knowledge_graph)
        generated_rules = rule_generator.generate_rules(
            target_relation=target_relation,
            n_rules=n_rules_pool
        )
        
        rule_pool = [
            RuleWithId(rule_id=f'rule_{i:03d}', rule=rule)
            for i, rule in enumerate(generated_rules)
        ]
    
    # ルールpoolを保存
    dir_iter_0_rules = os.path.join(dir_iter_0, 'rule_pool.pkl')
    import pickle
    with open(dir_iter_0_rules, 'wb') as fout:
        pickle.dump(rule_pool, fout)
    logger.info(f'  [OUTPUT] Rule pool saved to: {dir_iter_0_rules}')
    
    # ルール履歴の初期化
    logger.info('Step 0.3: Initializing rule history')
    rule_history = RuleHistory()
    logger.info(f'  [OUTPUT] Rule history initialized')
    
    # 評価器の初期化
    logger.info('Step 0.4: Initializing evaluator')
    evaluator = IterationEvaluator()
    logger.info(f'  [OUTPUT] Evaluator initialized')
    
    # 対象トリプルの読み込み
    logger.info('Step 0.5: Loading target triples')
    f_target_triples = os.path.join(dir_iter_0, 'target_triples.txt')
    with open(f_target_triples, 'r') as fin:
        all_target_triples = [tuple(line.strip().split('\t')) for line in fin]
    logger.info(f'  [OUTPUT] Loaded {len(all_target_triples)} target triples')
    
    # 現在のディレクトリを追跡
    dir_current = dir_iter_0
    
    # 【改善】対象トリプルスコア履歴（Early Stopping用）
    target_score_history = []
    
    # 反復フェーズ
    for iteration in range(1, n_iter + 1):
        logger.info('='*80)
        logger.info(f'Iteration {iteration}')
        logger.info('='*80)
        
        # 新しいiterationディレクトリ作成
        dir_next = os.path.join(dir_working, f'iter_{iteration}')
        os.makedirs(dir_next, exist_ok=True)
        
        # valid.txt, test.txtをコピー
        for fname in ['valid.txt', 'test.txt', 'target_triples.txt', 'config_dataset.json']:
            src = os.path.join(dir_current, fname)
            dst = os.path.join(dir_next, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)
        
        # 現在のモデルをロード
        logger.info('Step {}.0: Loading current model'.format(iteration))
        kge_before = KnowledgeGraphEmbedding(model_dir=dir_current)
        logger.info(f'  [INPUT] Model loaded from: {dir_current}')
        
        # ルール選択
        logger.info(f'Step {iteration}.1: Selecting {n_rules_select} rules from pool')
        rule_selector = create_rule_selector(
            strategy=rule_selector_strategy,
            rule_pool=rule_pool,
            rule_history=rule_history,
            llm_temperature=llm_temperature,
            iteration=iteration
        )
        
        selected_rules, _ = rule_selector.select_rules(rule_pool=rule_pool, k=n_rules_select, iteration=iteration)
        logger.info(f'  [OUTPUT] Selected {len(selected_rules)} rules')
        for i, rwid in enumerate(selected_rules):
            logger.info(f'    {i+1}. {rwid.rule_id}: {rwid.rule}')
        
        # トリプル追加（各ルールで独立したtarget triple セット）
        logger.info(f'Step {iteration}.2: Adding triples for each selected rule')
        
        all_added_triples = []
        successful_rules = []
        failed_rules = []
        used_target_triples = set()
        rule_evaluations = []
        
        for i, rule_with_id in enumerate(selected_rules):
            rule_id = rule_with_id.rule_id
            rule = rule_with_id.rule
            
            logger.info(f'  Processing rule: {rule_id}')
            logger.info(f'    [INPUT] Rule: {rule}')
            
            # このルール用のtarget tripleをサンプリング
            sampled_targets = sample_target_triples(
                all_target_triples,
                n_sample=n_targets_per_rule,
                exclude_triples=used_target_triples
            )
            
            logger.info(f'    [INPUT] Sampled {len(sampled_targets)} target triples for this rule')
            logger.info(f'    [INPUT] Already used target triples: {len(used_target_triples)}')
            
            # このルールでトリプル追加
            # 注：候補トリプルには新規エンティティが含まれるため、
            # 埋込モデルでのスコアリングは不可能。
            # 高品質なルールから抽出されたトリプルをそのまま使用。
            # 候補トリプルは常に初期データ（iter_0）にある
            dir_candidate = os.path.join(dir_working, 'iter_0')
            added_triples, added_info = add_triples_for_single_rule(
                dir_triples=dir_candidate,
                rule=rule,
                target_triples=sampled_targets
            )
            
            logger.info(f'    [OUTPUT] Added {len(added_triples)} triples for rule {rule_id}')
            
            if len(added_triples) > 0:
                all_added_triples.extend(added_triples)
                successful_rules.append((rule_id, rule, sampled_targets, added_triples))
                used_target_triples.update(sampled_targets)
            else:
                logger.warning(f'    [WARNING] Rule {rule_id} added 0 triples - marking as failed')
                failed_rules.append(rule_id)
                
                # ペナルティレコードを追加
                logger.info(f'    [PENALTY] Added penalty record for {rule_id} (mean_score=-10.0)')
                penalty_record = RuleEvaluationRecord(
                    iteration=iteration,
                    rule_id=rule_id,
                    rule=rule,
                    target_triples=sampled_targets,
                    added_triples=[],
                    score_changes=[],
                    mean_score_change=-10.0,
                    std_score_change=0.0,
                    positive_changes=0,
                    negative_changes=0
                )
                rule_history.add_record(penalty_record)
        
        # 失敗したルールの再選択
        max_reselection_attempts = 3
        for attempt in range(1, max_reselection_attempts + 1):
            if not failed_rules:
                break
            
            logger.info(f'  [RESELECTION] Attempt {attempt}: Reselecting rules for {len(failed_rules)} failed rules')
            
            # 再選択
            n_reselect = len(failed_rules)
            reselected_rules, _ = rule_selector.select_rules(rule_pool=rule_pool, k=n_reselect, iteration=iteration)
            logger.info(f'  [RESELECTION] Reselected {len(reselected_rules)} rules')
            for rwid in reselected_rules:
                logger.info(f'  [RESELECTION] Reselected rule: {rwid.rule_id}')
            
            new_failed = []
            for rule_with_id in reselected_rules:
                rule_id = rule_with_id.rule_id
                rule = rule_with_id.rule
                
                logger.info(f'  Processing rule: {rule_id}')
                logger.info(f'    [INPUT] Rule: {rule}')
                
                sampled_targets = sample_target_triples(
                    all_target_triples,
                    n_sample=n_targets_per_rule,
                    exclude_triples=used_target_triples
                )
                
                logger.info(f'    [INPUT] Sampled {len(sampled_targets)} target triples for this rule')
                logger.info(f'    [INPUT] Already used target triples: {len(used_target_triples)}')
                
                # 候補トリプルは常に初期データ（iter_0）にある
                dir_candidate = os.path.join(dir_working, 'iter_0')
                added_triples, added_info = add_triples_for_single_rule(
                    dir_triples=dir_candidate,
                    rule=rule,
                    target_triples=sampled_targets
                )
                
                logger.info(f'    [OUTPUT] Added {len(added_triples)} triples for rule {rule_id}')
                
                if len(added_triples) > 0:
                    all_added_triples.extend(added_triples)
                    successful_rules.append((rule_id, rule, sampled_targets, added_triples))
                    used_target_triples.update(sampled_targets)
                else:
                    logger.warning(f'    [WARNING] Rule {rule_id} added 0 triples - marking as failed')
                    new_failed.append(rule_id)
                    
                    logger.info(f'    [PENALTY] Added penalty record for {rule_id} (mean_score=-10.0)')
                    penalty_record = RuleEvaluationRecord(
                        iteration=iteration,
                        rule_id=rule_id,
                        rule=rule,
                        target_triples=sampled_targets,
                        added_triples=[],
                        score_changes=[],
                        mean_score_change=-10.0,
                        std_score_change=0.0,
                        positive_changes=0,
                        negative_changes=0
                    )
                    rule_history.add_record(penalty_record)
            
            failed_rules = new_failed
        
        logger.info(f'  [SUMMARY] Successful rules: {len(successful_rules)}, Failed rules: {len(failed_rules)}')
        
        # 重複を除いてユニークなトリプルのみ保持
        unique_added = list(set([tuple(t) for t in all_added_triples]))
        logger.info(f'  [OUTPUT] Total unique triples added across all rules: {len(unique_added)}')
        
        # train.txtを更新
        logger.info(f'  [INPUT] Original train file: {os.path.join(dir_current, "train.txt")}')
        
        # 元のtrain.txtを読み込み
        train_factory = TriplesFactory.from_path(
            path=os.path.join(dir_current, 'train.txt'),
            create_inverse_triples=False
        )
        original_triples = [tuple(t) for t in train_factory.triples.tolist()]
        logger.info(f'  [INPUT] Original train triples: {len(original_triples)}')
        
        # 新しいトリプルを追加
        updated_triples = original_triples + unique_added
        logger.info(f'  [OUTPUT] Updated train triples: {len(updated_triples)} (+{len(unique_added)})')
        
        # 更新されたtrain.txtを保存
        f_train_next = os.path.join(dir_next, 'train.txt')
        with open(f_train_next, 'w') as fout:
            for h, r, t in updated_triples:
                fout.write(f'{h}\t{r}\t{t}\n')
        
        logger.info(f'  [OUTPUT] Updated train file saved to: {f_train_next}')
        
        # 追加詳細をJSON保存
        rule_additions = {
            'iteration': iteration,
            'rules': [
                {
                    'rule_id': rule_id,
                    'rule': str(rule),
                    'n_target_triples': len(targets),
                    'n_added_triples': len(added)
                }
                for rule_id, rule, targets, added in successful_rules
            ],
            'total_added': len(unique_added)
        }
        
        f_rule_additions = os.path.join(dir_next, 'rule_additions.json')
        with open(f_rule_additions, 'w') as fout:
            json.dump(rule_additions, fout, indent=2)
        
        logger.info(f'  [OUTPUT] Rule additions details saved to: {f_rule_additions}')
        
        # 更新されたモデルを学習
        logger.info(f'Step {iteration}.3: Training embedding model with all added triples')
        logger.info(f'  [INPUT] Training data directory: {dir_next}')
        logger.info(f'  [INPUT] Model: {config_embedding["model"]}')
        logger.info(f'  [INPUT] Embedding dim: {config_embedding.get("model_kwargs", {}).get("embedding_dim", "N/A")}')
        logger.info(f'  [INPUT] Epochs: {num_epochs}')
        
        config_embedding['dir_triples'] = dir_next
        config_embedding['dir_save'] = dir_next
        kge_after = KnowledgeGraphEmbedding.train_model(**config_embedding)
        
        logger.info(f'  [OUTPUT] Updated model saved to: {dir_next}')
        logger.info(f'  [OUTPUT] Model type: {type(kge_after).__name__}')
        
        # ルール評価と履歴記録
        logger.info(f'Step {iteration}.4: Evaluating each rule and recording to history')
        logger.info(f'  [INPUT] Model before: {type(kge_before).__name__}')
        logger.info(f'  [INPUT] Model after: {type(kge_after).__name__}')
        logger.info(f'  [INPUT] Number of successful rules to evaluate: {len(successful_rules)}')
        
        for rule_id, rule, targets, added in successful_rules:
            logger.info(f'  Evaluating rule: {rule_id}')
            logger.info(f'    [INPUT] Target triples: {len(targets)}')
            logger.info(f'    [INPUT] Added triples: {len(added)}')
            
            # RuleWiseAnalyzerで評価レコードを作成
            analyzer = RuleWiseAnalyzer(
                kge_before=kge_before,
                kge_after=kge_after
            )
            
            record = analyzer.create_evaluation_record(
                iteration=iteration,
                rule_id=rule_id,
                rule=rule,
                target_triples=targets,
                added_triples=added
            )
            
            logger.info(f'    [OUTPUT] Score change mean: {record.mean_score_change:.6f}')
            logger.info(f'    [OUTPUT] Score change std: {record.std_score_change:.6f}')
            logger.info(f'    [OUTPUT] Positive changes: {record.positive_changes}')
            logger.info(f'    [OUTPUT] Negative changes: {record.negative_changes}')
            
            # 履歴に記録
            rule_history.add_record(record)
        
        # 履歴を保存
        f_history = os.path.join(dir_next, 'rule_history.pkl')
        rule_history.save(f_history)
        logger.info(f'  [OUTPUT] Rule history saved to: {f_history}')
        
        f_history_json = os.path.join(dir_next, 'rule_history.json')
        rule_history.save_json(f_history_json)
        logger.info(f'  [OUTPUT] Rule history (JSON) saved to: {f_history_json}')
        
        logger.info(f'  [OUTPUT] Total records in history: {len(rule_history.records)}')
        
        # サマリーレポート生成
        md_summary = rule_history.generate_summary_report()
        f_summary = os.path.join(dir_next, 'rule_history_summary.md')
        save_markdown(md_summary, f_summary)
        logger.info(f'  [OUTPUT] History summary report saved to: {f_summary}')
        
        # Iteration評価
        logger.info(f'Step {iteration}.5: Evaluating iteration performance')
        logger.info(f'  [INPUT] Evaluating {len(all_target_triples)} target triples')
        
        iteration_metrics = evaluator.evaluate_iteration(
            iteration=iteration,
            kge_before=kge_before,
            kge_after=kge_after,
            target_triples=all_target_triples,
            n_triples_added=len(unique_added),
            dir_save=dir_next
        )
        
        logger.info(f'  [OUTPUT] Iteration evaluation saved to: {dir_next}')
        
        # 【改善】スコア履歴に追加
        target_score_history.append(iteration_metrics.target_score_after)
        
        # 【改善】Early Stopping判定
        if not check_iteration_quality(target_score_history, patience=early_stop_patience):
            logger.warning('='*80)
            logger.warning('EARLY STOPPING: Target score degradation detected')
            logger.warning('='*80)
            break
        
        # ルールpoolは固定（多腕バンディット戦略）
        logger.info(f'Step {iteration}.6: Rule pool remains fixed (multi-armed bandit arms)')
        logger.info(f'  [INFO] Rule pool size: {len(rule_pool)} (unchanged)')
        logger.info(f'  [INFO] Multi-armed bandit strategy: arms are not modified, only selection improves')
        
        # 次のiteration用にルールpoolをコピー保存（記録のため）
        f_pool = os.path.join(dir_next, 'rule_pool.pkl')
        with open(f_pool, 'wb') as fout:
            pickle.dump(rule_pool, fout)
        logger.info(f'  [OUTPUT] Rule pool (copy) saved to: {f_pool}')
        
        # CSVで保存
        import pandas as pd
        pool_data = []
        for rwid in rule_pool:
            pool_data.append({
                'rule_id': rwid.rule_id,
                'head': str(rwid.rule.head),
                'body': ' AND '.join([str(bp) for bp in rwid.rule.body])
            })
        df_pool = pd.DataFrame(pool_data)
        f_pool_csv = os.path.join(dir_next, 'rule_pool.csv')
        df_pool.to_csv(f_pool_csv, index=False)
        logger.info(f'  [OUTPUT] Rule pool (CSV) saved to: {f_pool_csv}')
        
        # 次のiterationの準備
        dir_current = dir_next
        logger.info(f'  [STATE] Current directory updated to: {dir_current}')
        
        logger.info(f'Iteration {iteration} completed\n')
    
    # 最終レポート生成
    logger.info('='*80)
    logger.info('All iterations completed!')
    logger.info('='*80)
    
    logger.info('Generating final evaluation report')
    evaluator.create_final_report(dir_working)
    logger.info(f'Final evaluation report saved to: {dir_working}')
    
    logger.info(f'Results saved in: {dir_working}')
