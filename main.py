

"""
メインスクリプト - 新アルゴリズム版

多腕バンディット戦略を用いたルール選択による知識グラフ改善。

アルゴリズムフロー:
0. 初期化フェーズ:
   - 埋込モデル学習
   - AMIE+ルール抽出
   - LLMでルールpool生成（15-20個）
   - ルール履歴初期化

1. 反復フェーズ (各iteration):
   a. ルール選択（UCBアルゴリズム）: poolからk個選択
   b. 各選択ルールで独立したtarget tripleセットに対してトリプル追加
   c. 全トリプル追加後に埋込モデル学習（1回）
   d. 各ルールのスコア変化分析と履歴記録
   e. 履歴全体を使ってルールpool更新
"""

import json
import os
import random
import shutil

from simple_active_refine.rule_generator import BaseRuleGenerator
from simple_active_refine.embedding import KnowledgeGraphEmbedding
from simple_active_refine.triples_editor import add_triples_for_single_rule
from simple_active_refine.analyzer import RuleWiseAnalyzer
from simple_active_refine.rule_extractor import extract_rules_from_high_score_triples
from simple_active_refine.rule_history import RuleHistory
from simple_active_refine.rule_selector import create_rule_selector, RuleWithId
from simple_active_refine.util import get_logger
from pykeen.triples import TriplesFactory

logger = get_logger('main')


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


if __name__ == '__main__':
    """パラメータ設定"""
    # 対象知識グラフとリレーション
    knowledge_graph = 'FB15k-237'
    target_relation = '/location/location/contains'
    
    # 全体制御
    dir_initial_triples = './experiments/test_data_for_locations_contain'
    n_iter = 10
    dir_working = './experiments/20251130/new_algorithm_test1'
    
    # ルールpool設定
    n_rules_pool = 15  # ルールpool内のルール数
    n_rules_select = 3  # 各iterationで選択するルール数
    n_targets_per_rule = 10  # 各ルールあたりのtarget triple数
    
    # ルール選択戦略（LLMポリシー駆動）
    rule_selector_strategy = 'llm_policy'  # 'llm_policy', 'ucb', 'epsilon_greedy'
    llm_temperature = 0.3  # LLMの創造性（0-1、低いほど決定論的）
    
    # AMIE+設定
    use_amie_rules = True
    min_head_coverage = 0.01
    min_pca_conf = 0.1
    k_neighbor = 1
    lower_percentile = 80
    
    # 埋込モデル設定
    f_config_embedding = "./config_embeddings.json"
    num_epochs = 100
    
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
    logger.info('Phase 0: Initialization')
    logger.info('='*80)
    
    # Iteration 0の作成
    dir_iter_0 = os.path.join(dir_working, 'iter_0')
    if os.path.exists(dir_iter_0):
        logger.info(f'Removing existing directory {dir_iter_0}')
        shutil.rmtree(dir_iter_0)
    shutil.copytree(dir_initial_triples, dir_iter_0)
    
    # 初期埋込モデル学習
    logger.info('Step 0.1: Training initial knowledge graph embedding')
    config_embedding['dir_triples'] = dir_iter_0
    config_embedding['dir_save'] = dir_iter_0
    kge_initial = KnowledgeGraphEmbedding.train_model(**config_embedding)
    
    # AMIE+ルール抽出（参考用）
    logger.info('Step 0.2: Extracting AMIE+ rules for reference')
    if use_amie_rules:
        amie_rules = extract_rules_from_high_score_triples(
            kge_initial,
            config_dataset['target_relation'],
            min_pca_conf=min_pca_conf,
            min_head_coverage=min_head_coverage,
            k_neighbor=k_neighbor,
            lower_percentile=lower_percentile
        )
        amie_rules.to_csv(os.path.join(dir_iter_0, 'amie_rules.csv'))
    else:
        amie_rules = None
    
    # 初期ルールpool生成
    logger.info(f'Step 0.3: Generating initial rule pool ({n_rules_pool} rules)')
    rule_generator = BaseRuleGenerator()
    rule_pool_amie = rule_generator.generate_initial_rule_pool(
        knowledge_graph,
        target_relation,
        n_rules=n_rules_pool,
        ref_rules=amie_rules
    )
    
    # ルールにIDを付与
    rule_pool = []
    for i, rule in enumerate(rule_pool_amie.rules):
        rule_id = f"rule_{i:03d}"
        # metadataにIDを保存
        if rule.metadata is None:
            rule.metadata = {}
        rule.metadata['rule_id'] = rule_id
        rule_pool.append(RuleWithId(rule_id=rule_id, rule=rule))
    
    logger.info(f'Created rule pool with {len(rule_pool)} rules')
    
    # ルール履歴初期化
    logger.info('Step 0.4: Initializing rule history')
    rule_history = RuleHistory()
    
    # ルール選択器初期化
    rule_selector = create_rule_selector(
        strategy=rule_selector_strategy,
        history=rule_history,
        temperature=llm_temperature
    )
    
    # 全target tripleの読み込み
    f_target_triples = os.path.join(dir_iter_0, 'target_triples.txt')
    all_target_triples = []
    with open(f_target_triples, 'r') as fin:
        for row in fin:
            words = row.rstrip().split('\t')
            all_target_triples.append((words[0], words[1], words[2]))
    
    logger.info(f'Total target triples available: {len(all_target_triples)}')
    
    # 初期ルールpoolを保存
    rule_pool_amie.to_pickle(os.path.join(dir_iter_0, 'rule_pool.pkl'))
    rule_pool_amie.to_csv(os.path.join(dir_iter_0, 'rule_pool.csv'))
    
    # 現在のディレクトリ
    dir_current = dir_iter_0
    kge_current = kge_initial
    
    logger.info('='*80)
    logger.info('Phase 1: Iterative Improvement')
    logger.info('='*80)
    
    for i in range(1, n_iter + 1):
        logger.info(f'\n{"="*80}')
        logger.info(f'Iteration {i}/{n_iter}')
        logger.info(f'{"="*80}')
        
        # 次iteration用のディレクトリを事前作成（ポリシー保存用）
        dir_next = os.path.join(dir_working, f'iter_{i}')
        if os.path.exists(dir_next):
            logger.info(f'Removing existing directory {dir_next}')
            shutil.rmtree(dir_next)
        os.makedirs(dir_next, exist_ok=True)
        
        # Step 1: ルール選択
        logger.info(f'Step {i}.1: Selecting {n_rules_select} rules from pool using {rule_selector_strategy}')
        selected_rules, updated_policy = rule_selector.select_rules(rule_pool, k=n_rules_select, iteration=i)
        
        # ポリシーを保存
        if updated_policy:
            save_markdown(f"# Selection Policy (Iteration {i})\n\n{updated_policy}",
                         os.path.join(dir_next, f'selection_policy_iter{i}.md'))
        
        for j, rule_with_id in enumerate(selected_rules, 1):
            logger.info(f'  Selected rule {j}: {rule_with_id.rule_id}')
        
        # Step 2: 各ルールでトリプル追加（累積）
        logger.info(f'Step {i}.2: Adding triples for each selected rule')
        
        added_triples_by_rule = {}
        used_target_triples = set()
        
        # 各ルール用のトリプル追加を累積的に実行
        for rule_with_id in selected_rules:
            rule_id = rule_with_id.rule_id
            logger.info(f'  Processing rule: {rule_id}')
            
            # このルール用のtarget tripleをサンプリング
            target_triples_for_rule = sample_target_triples(
                all_target_triples,
                n_targets_per_rule,
                exclude_triples=used_target_triples
            )
            used_target_triples.update(target_triples_for_rule)
            
            logger.info(f'    Sampled {len(target_triples_for_rule)} target triples for this rule')
            
            # トリプル追加
            added_triples, details = add_triples_for_single_rule(
                dir_triples=dir_current,
                rule=rule_with_id.rule,
                target_triples=target_triples_for_rule
            )
            
            added_triples_by_rule[rule_id] = {
                'rule': rule_with_id.rule,
                'target_triples': target_triples_for_rule,
                'added_triples': added_triples,
                'details': details
            }
            
            logger.info(f'    Added {len(added_triples)} triples')
        
        # 既存ファイルのコピー（dir_nextは既に作成済み）
        for filename in ['test.txt', 'valid.txt', 'train_removed.txt',
                        'test_removed.txt', 'valid_removed.txt',
                        'config_dataset.json', 'target_triples.txt']:
            f_src = os.path.join(dir_current, filename)
            f_dst = os.path.join(dir_next, filename)
            if os.path.exists(f_src):
                shutil.copy(f_src, f_dst)
        
        # 全ルールで追加されたトリプルをマージ
        all_added_triples = set()
        for rule_id, rule_data in added_triples_by_rule.items():
            all_added_triples.update(rule_data['added_triples'])
        
        # 元のtrainトリプルと結合
        tf_org = TriplesFactory.from_path(os.path.join(dir_current, 'train.txt'))
        set_org_triples = set(map(tuple, tf_org.triples.tolist()))
        set_updated_triples = set_org_triples | all_added_triples
        
        logger.info(f'Total triples added: {len(all_added_triples)}')
        logger.info(f'Updated train set: {len(set_org_triples)} -> {len(set_updated_triples)}')
        
        # 更新されたtrainファイルを保存
        with open(os.path.join(dir_next, 'train.txt'), 'w') as fout:
            for h, r, t in set_updated_triples:
                fout.write(f'{h}\t{r}\t{t}\n')
        
        # 各ルールの追加詳細をJSON保存
        rule_additions_info = {}
        for rule_id, rule_data in added_triples_by_rule.items():
            rule_additions_info[rule_id] = {
                'target_triples': [list(t) for t in rule_data['target_triples']],
                'added_triples': [list(t) for t in rule_data['added_triples']],
                'n_targets': len(rule_data['target_triples']),
                'n_added': len(rule_data['added_triples'])
            }
        
        with open(os.path.join(dir_next, 'rule_additions.json'), 'w') as fout:
            json.dump(rule_additions_info, fout, indent=2)
        
        # Step 3: 埋込モデル学習（全トリプル追加後に1回）
        logger.info(f'Step {i}.3: Training embedding model with all added triples')
        config_embedding['dir_triples'] = dir_next
        config_embedding['dir_save'] = dir_next
        kge_next = KnowledgeGraphEmbedding.train_model(**config_embedding)
        
        # Step 4: 各ルールの評価と履歴記録
        logger.info(f'Step {i}.4: Evaluating each rule and recording to history')
        analyzer = RuleWiseAnalyzer(kge_before=kge_current, kge_after=kge_next)
        
        for rule_with_id in selected_rules:
            rule_id = rule_with_id.rule_id
            rule_data = added_triples_by_rule[rule_id]
            
            evaluation_record = analyzer.create_evaluation_record(
                iteration=i,
                rule_id=rule_id,
                rule=rule_data['rule'],
                target_triples=rule_data['target_triples'],
                added_triples=rule_data['added_triples']
            )
            
            rule_history.add_record(evaluation_record)
        
        # 履歴を保存
        rule_history.save(os.path.join(dir_next, 'rule_history.pkl'))
        rule_history.save_json(os.path.join(dir_next, 'rule_history.json'))
        
        # 履歴サマリーレポート生成
        summary_report = rule_history.generate_summary_report()
        save_markdown(summary_report, os.path.join(dir_next, 'rule_history_summary.md'))
        
        # Step 5: ルールpool更新
        logger.info(f'Step {i}.5: Updating rule pool based on history')
        
        # AMIE+ルールを再抽出（参考用）
        if use_amie_rules:
            amie_rules = extract_rules_from_high_score_triples(
                kge_next,
                config_dataset['target_relation'],
                min_pca_conf=min_pca_conf,
                min_head_coverage=min_head_coverage,
                k_neighbor=k_neighbor,
                lower_percentile=lower_percentile
            )
            amie_rules.to_csv(os.path.join(dir_next, 'amie_rules.csv'))
        
        # ルールpoolを更新
        updated_pool_amie = rule_generator.update_rule_pool_with_history(
            knowledge_graph=knowledge_graph,
            target_relation=target_relation,
            current_pool=rule_pool_amie,
            history=rule_history,
            n_keep_best=10,
            n_generate_new=5,
            ref_rules=amie_rules
        )
        
        # 新しいルールにIDを付与
        new_rule_pool = []
        for rule in updated_pool_amie.rules:
            # 既存のIDがあればそれを使用、なければ新規生成
            if rule.metadata and 'rule_id' in rule.metadata:
                rule_id = rule.metadata['rule_id']
            else:
                rule_id = f"rule_new_{i}_{len(new_rule_pool)}"
                if rule.metadata is None:
                    rule.metadata = {}
                rule.metadata['rule_id'] = rule_id
            new_rule_pool.append(RuleWithId(rule_id=rule_id, rule=rule))
        
        rule_pool = new_rule_pool
        rule_pool_amie = updated_pool_amie
        
        # 更新されたルールpoolを保存
        rule_pool_amie.to_pickle(os.path.join(dir_next, 'rule_pool.pkl'))
        rule_pool_amie.to_csv(os.path.join(dir_next, 'rule_pool.csv'))
        
        logger.info(f'Updated rule pool: {len(rule_pool)} rules')
        
        # 次iterationの準備
        dir_current = dir_next
        kge_current = kge_next
        
        logger.info(f'Iteration {i} completed\n')
    
    logger.info('='*80)
    logger.info('All iterations completed!')
    logger.info('='*80)
    
    # 最終レポート生成
    final_report = rule_history.generate_summary_report()
    save_markdown(final_report, os.path.join(dir_working, 'final_rule_history_summary.md'))
    
    logger.info(f'Results saved in: {dir_working}')
