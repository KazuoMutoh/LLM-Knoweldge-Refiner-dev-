

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
from simple_active_refine.rule_extractor import extract_rules_from_high_score_triples, extract_rules_from_entire_graph
from simple_active_refine.rule_history import RuleHistory
from simple_active_refine.rule_selector import create_rule_selector, RuleWithId
from simple_active_refine.evaluation import IterationEvaluator
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
    target_relation = '/people/person/nationality'
    
    # 全体制御
    dir_initial_triples = './experiments/test_data_for_nationality_v3'  # v3: min_target_triples=5, 相互近傍除外, 自動リフィル
    n_iter = 2  # テスト用: 評価機能のテスト（想定実行時間: 5-8分）
    dir_working = './experiments/20251214/try2'
    
    # ルールpool設定（効率型）
    n_rules_pool = 12      # AMIE+上位12個（73個中）
    n_rules_select = 3     # 各iterationで3個選択（poolの25%）
    n_targets_per_rule = 10  # 統計的に十分な評価サンプル
    
    # ルール選択戦略（LLMポリシー駆動）
    rule_selector_strategy = 'llm_policy'  # 'llm_policy', 'ucb', 'epsilon_greedy'
    llm_temperature = 0.3  # LLMの創造性（0-1、低いほど決定論的）
    
    # AMIE+設定
    use_amie_rules = True  # Javaがインストールされたので有効化
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
    # 初期データセット（train.txt, valid.txt, test.txt）から知識グラフ埋込モデル（TransE等）を学習。
    # 学習したモデルは、トリプルの尤もらしさをスコア化するために使用され、後のステップでルール抽出や
    # トリプル追加の効果測定に活用される。
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
    
    # AMIE+ルール抽出（初期ルールpool用）
    # 知識グラフ全体からAMIE+を用いてHornルールを抽出。
    # これらのルールを直接初期ルールプールとして使用する。
    # bodyに/people/person/nationalityを含むルールも許可し、実データに基づく有望なパターンを活用する。
    logger.info('Step 0.2: Extracting AMIE+ rules for initial rule pool')
    if use_amie_rules:
        logger.info(f'  [INPUT] Target relation: {config_dataset["target_relation"]}')
        logger.info(f'  [INPUT] Min PCA confidence: {min_pca_conf}')
        logger.info(f'  [INPUT] Min head coverage: {min_head_coverage}')
        logger.info(f'  [INPUT] Extraction method: entire knowledge graph')
        amie_rules = extract_rules_from_entire_graph(
            kge_initial,
            config_dataset['target_relation'],
            min_pca_conf=min_pca_conf,
            min_head_coverage=min_head_coverage,
            top_k=100  # Extract more rules for better selection
        )
        amie_rules.to_csv(os.path.join(dir_iter_0, 'amie_rules.csv'))
        logger.info(f'  [OUTPUT] Number of AMIE+ rules extracted: {len(amie_rules.rules)}')
        logger.info(f'  [OUTPUT] Rules saved to: {os.path.join(dir_iter_0, "amie_rules.csv")}')
    else:
        amie_rules = None
        logger.info('  [OUTPUT] AMIE+ rules extraction skipped')
    
    # 初期ルールpool生成（AMIE+ルールから選択）
    # AMIE+で抽出されたルールから、pca_conf値などの品質指標に基づいて上位n_rules_pool個を選択。
    # LLMによる生成ではなく、実データに基づくAMIE+ルールを直接使用することで、
    # 知識グラフ内に実際に存在するパターンとのマッチング率を向上させる。
    logger.info(f'Step 0.3: Creating initial rule pool from AMIE+ rules (top {n_rules_pool} by pca_conf)')
    logger.info(f'  [INPUT] Available AMIE+ rules: {len(amie_rules.rules) if amie_rules else 0}')
    logger.info(f'  [INPUT] Number of rules to select: {n_rules_pool}')
    logger.info(f'  [INPUT] Selection criterion: pca_conf (confidence)')
    rule_generator = BaseRuleGenerator()
    
    if amie_rules and len(amie_rules.rules) > 0:
        # AMIE+ルールから直接選択
        rule_pool_amie = rule_generator.create_initial_rule_pool_from_amie(
            amie_rules=amie_rules,
            n_rules=n_rules_pool,
            sort_by='pca_conf'  # 信頼度が高いルールを優先
        )
    else:
        # フォールバック: AMIE+ルールがない場合はLLMで生成
        logger.warning('  [WARNING] No AMIE+ rules available, falling back to LLM generation')
        rule_pool_amie = rule_generator.generate_initial_rule_pool(
            knowledge_graph,
            target_relation,
            n_rules=n_rules_pool,
            ref_rules=amie_rules
        )
    
    logger.info(f'  [OUTPUT] Number of rules in initial pool: {len(rule_pool_amie.rules)}')
    
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
    logger.info(f'  [OUTPUT] Rule pool saved to: {os.path.join(dir_iter_0, "rule_pool.pkl")}')
    
    # ルール履歴初期化
    # 各ルールの使用履歴（どのiterationで選択され、どれだけスコアが改善したか）を管理する
    # RuleHistoryオブジェクトを初期化。この履歴情報は、ルール選択戦略（UCB, LLMポリシー等）と
    # ルールpool更新の判断材料として活用される。
    logger.info('Step 0.4: Initializing rule history')
    rule_history = RuleHistory()
    logger.info(f'  [OUTPUT] Rule history initialized (empty)')
    
    # 評価器初期化
    # 各iterationでの対象トリプルのスコア変化、追加トリプル数、知識グラフ埋め込み全体の精度を
    # 記録・分析する評価器を初期化。最終的に全iterationの結果をまとめたレポートを生成する。
    logger.info('Step 0.5: Initializing iteration evaluator')
    evaluator = IterationEvaluator()
    logger.info(f'  [OUTPUT] Iteration evaluator initialized')
    
    # ルール選択器初期化
    # 指定された戦略（'llm_policy', 'ucb', 'epsilon_greedy'）に基づいてルール選択器を作成。
    # LLMポリシー戦略の場合、履歴情報を解析してLLMが選択判断を行う。
    logger.info(f'  [INPUT] Rule selector strategy: {rule_selector_strategy}')
    logger.info(f'  [INPUT] LLM temperature: {llm_temperature}')
    rule_selector = create_rule_selector(
        strategy=rule_selector_strategy,
        history=rule_history,
        temperature=llm_temperature
    )
    logger.info(f'  [OUTPUT] Rule selector created: {type(rule_selector).__name__}')
    
    # 全target tripleの読み込み
    f_target_triples = os.path.join(dir_iter_0, 'target_triples.txt')
    logger.info(f'  [INPUT] Loading target triples from: {f_target_triples}')
    all_target_triples = []
    with open(f_target_triples, 'r') as fin:
        for row in fin:
            words = row.rstrip().split('\t')
            all_target_triples.append((words[0], words[1], words[2]))
    
    logger.info(f'  [OUTPUT] Total target triples available: {len(all_target_triples)}')
    
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
        # ルールpoolから、選択戦略（UCB, LLMポリシー等）に基づいてn_rules_select個のルールを選択。
        # 各ルールの過去の成績（スコア改善度）を考慮しつつ、探索と活用のバランスを取る。
        # LLMポリシー戦略の場合、履歴を分析して選択理由も生成される。
        logger.info(f'Step {i}.1: Selecting {n_rules_select} rules from pool using {rule_selector_strategy}')
        logger.info(f'  [INPUT] Rule pool size: {len(rule_pool)}')
        logger.info(f'  [INPUT] Number of rules to select: {n_rules_select}')
        logger.info(f'  [INPUT] Selection strategy: {rule_selector_strategy}')
        logger.info(f'  [INPUT] Current iteration: {i}')
        logger.info(f'  [DEBUG] History records before selection: {len(rule_history.records)}')
        logger.info(f'  [DEBUG] Selector history records: {len(rule_selector.history.records)}')
        logger.info(f'  [DEBUG] Same history object? {rule_history is rule_selector.history}')
        selected_rules, updated_policy = rule_selector.select_rules(rule_pool, k=n_rules_select, iteration=i)
        
        logger.info(f'  [OUTPUT] Number of rules selected: {len(selected_rules)}')
        for j, rule_with_id in enumerate(selected_rules, 1):
            logger.info(f'  [OUTPUT] Selected rule {j}: {rule_with_id.rule_id}')
        
        # ポリシーを保存
        if updated_policy:
            policy_file = os.path.join(dir_next, f'selection_policy_iter{i}.md')
            save_markdown(f"# Selection Policy (Iteration {i})\n\n{updated_policy}", policy_file)
            logger.info(f'  [OUTPUT] Selection policy saved to: {policy_file}')
        
        # Step 2: 各ルールでトリプル追加（累積）
        # 選択された各ルールに対して、重複しないtarget tripleセットをサンプリングし、
        # ルールのbodyパターンにマッチするトリプルを探索。マッチしたパターンから新しいトリプルを
        # 推論・追加する。各ルールの追加トリプルは個別に記録され、後で効果測定に使用される。
        # トリプルが追加されなかったルールは再選択の対象とする。
        logger.info(f'Step {i}.2: Adding triples for each selected rule')
        logger.info(f'  [INPUT] Current data directory: {dir_current}')
        logger.info(f'  [INPUT] Total target triples available: {len(all_target_triples)}')
        logger.info(f'  [INPUT] Targets per rule: {n_targets_per_rule}')
        
        added_triples_by_rule = {}
        used_target_triples = set()
        failed_rules = []  # トリプルが追加されなかったルール
        successful_rules = []  # トリプルが追加されたルール
        
        # 選択されたルールのコピー（再選択に使用）
        remaining_rules = list(selected_rules)
        max_reselection_attempts = 3  # 最大再選択回数
        
        for attempt in range(max_reselection_attempts):
            if not remaining_rules:
                break
                
            if attempt > 0:
                logger.info(f'  [RESELECTION] Attempt {attempt + 1}: Reselecting rules for {len(remaining_rules)} failed rules')
            
            rules_to_process = list(remaining_rules)
            remaining_rules = []
            
            # 各ルール用のトリプル追加を累積的に実行
            for rule_with_id in rules_to_process:
                rule_id = rule_with_id.rule_id
                logger.info(f'  Processing rule: {rule_id}')
                logger.info(f'    [INPUT] Rule: {rule_with_id.rule.head} <- {[str(b) for b in rule_with_id.rule.body]}')
                
                # このルール用のtarget tripleをサンプリング
                target_triples_for_rule = sample_target_triples(
                    all_target_triples,
                    n_targets_per_rule,
                    exclude_triples=used_target_triples
                )
                used_target_triples.update(target_triples_for_rule)
                
                logger.info(f'    [INPUT] Sampled {len(target_triples_for_rule)} target triples for this rule')
                logger.info(f'    [INPUT] Already used target triples: {len(used_target_triples)}')
                
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
                
                logger.info(f'    [OUTPUT] Added {len(added_triples)} triples for rule {rule_id}')
                
                # トリプルが追加されなかった場合
                if len(added_triples) == 0:
                    logger.warning(f'    [WARNING] Rule {rule_id} added 0 triples - marking as failed')
                    failed_rules.append(rule_with_id)
                    # 低いスコアのペナルティ記録を作成
                    from simple_active_refine.rule_history import RuleEvaluationRecord
                    penalty_record = RuleEvaluationRecord(
                        iteration=i,
                        rule_id=rule_id,
                        rule=rule_with_id.rule,
                        target_triples=target_triples_for_rule,
                        added_triples=[],
                        score_changes=[0.0] * len(target_triples_for_rule),
                        mean_score_change=-10.0,  # 大きなペナルティ
                        std_score_change=0.0,
                        positive_changes=0,
                        negative_changes=len(target_triples_for_rule)
                    )
                    rule_history.add_record(penalty_record)
                    logger.info(f'    [PENALTY] Added penalty record for {rule_id} (mean_score=-10.0)')
                else:
                    successful_rules.append(rule_with_id)
            
            # 失敗したルールがある場合、再選択
            if failed_rules and attempt < max_reselection_attempts - 1:
                # 失敗したルールと成功したルールを除外して再選択
                excluded_rule_ids = {r.rule_id for r in failed_rules + successful_rules}
                available_for_reselection = [r for r in rule_pool if r.rule_id not in excluded_rule_ids]
                
                if available_for_reselection:
                    n_reselect = min(len(failed_rules), len(available_for_reselection))
                    logger.info(f'  [RESELECTION] Selecting {n_reselect} replacement rules from {len(available_for_reselection)} available rules')
                    
                    # iter=0の場合、pca_confでソート選択（LLMを使わない）
                    if i == 0:
                        logger.info(f'  [RESELECTION] Iteration 0: Selecting by pca_conf (AMIE+ confidence)')
                        sorted_available = sorted(
                            available_for_reselection,
                            key=lambda r: r.rule.pca_conf if r.rule.pca_conf is not None else -1,
                            reverse=True
                        )
                        reselected = sorted_available[:n_reselect]
                        for rule_with_id in reselected:
                            pca_conf = rule_with_id.rule.pca_conf if rule_with_id.rule.pca_conf is not None else 0.0
                            logger.info(f'  [RESELECTION] Reselected rule: {rule_with_id.rule_id} (pca_conf={pca_conf:.4f})')
                    else:
                        # 再選択（履歴を考慮）
                        reselected, _ = rule_selector.select_rules(available_for_reselection, k=n_reselect, iteration=i)
                        for r in reselected:
                            logger.info(f'  [RESELECTION] Reselected rule: {r.rule_id}')
                    
                    remaining_rules = reselected
                else:
                    logger.warning(f'  [RESELECTION] No more rules available for reselection')
                    break
                
                failed_rules = []  # リセット
            else:
                break
        
        # 最終的な結果をログ
        logger.info(f'  [SUMMARY] Successful rules: {len(successful_rules)}, Failed rules: {len(failed_rules)}')
        
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
        
        logger.info(f'  [OUTPUT] Total unique triples added across all rules: {len(all_added_triples)}')
        
        # 元のtrainトリプルと結合
        train_file = os.path.join(dir_current, 'train.txt')
        logger.info(f'  [INPUT] Original train file: {train_file}')
        tf_org = TriplesFactory.from_path(train_file)
        set_org_triples = set(map(tuple, tf_org.triples.tolist()))
        set_updated_triples = set_org_triples | all_added_triples
        
        logger.info(f'  [INPUT] Original train triples: {len(set_org_triples)}')
        logger.info(f'  [OUTPUT] Updated train triples: {len(set_updated_triples)} (+{len(set_updated_triples) - len(set_org_triples)})')
        
        # 更新されたtrainファイルを保存
        updated_train_file = os.path.join(dir_next, 'train.txt')
        with open(updated_train_file, 'w') as fout:
            for h, r, t in set_updated_triples:
                fout.write(f'{h}\t{r}\t{t}\n')
        logger.info(f'  [OUTPUT] Updated train file saved to: {updated_train_file}')
        
        # 各ルールの追加詳細をJSON保存
        rule_additions_info = {}
        for rule_id, rule_data in added_triples_by_rule.items():
            rule_additions_info[rule_id] = {
                'target_triples': [list(t) for t in rule_data['target_triples']],
                'added_triples': [list(t) for t in rule_data['added_triples']],
                'n_targets': len(rule_data['target_triples']),
                'n_added': len(rule_data['added_triples'])
            }
        
        rule_additions_file = os.path.join(dir_next, 'rule_additions.json')
        with open(rule_additions_file, 'w') as fout:
            json.dump(rule_additions_info, fout, indent=2)
        logger.info(f'  [OUTPUT] Rule additions details saved to: {rule_additions_file}')
        
        # Step 3: 埋込モデル学習（全トリプル追加後に1回）
        # 全ルールで追加されたトリプルを含む更新されたデータセットで埋込モデルを再学習。
        # この新しいモデルを使って、各target tripleのスコア変化を計算し、どのルールが効果的だったかを
        # 評価する。効率化のため、複数ルールのトリプル追加後に1回だけ学習を実行する。
        logger.info(f'Step {i}.3: Training embedding model with all added triples')
        logger.info(f'  [INPUT] Training data directory: {dir_next}')
        logger.info(f'  [INPUT] Model: {config_embedding["model"]}')
        logger.info(f'  [INPUT] Embedding dim: {config_embedding.get("model_kwargs", {}).get("embedding_dim", "N/A")}')
        logger.info(f'  [INPUT] Epochs: {num_epochs}')
        config_embedding['dir_triples'] = dir_next
        config_embedding['dir_save'] = dir_next
        kge_next = KnowledgeGraphEmbedding.train_model(**config_embedding)
        logger.info(f'  [OUTPUT] Updated model saved to: {dir_next}')
        logger.info(f'  [OUTPUT] Model type: {type(kge_next).__name__}')
        
        # Step 4: 各ルールの評価と履歴記録
        # トリプル追加前後の埋込モデルを比較し、各ルールのtarget tripleスコア変化を分析。
        # スコア改善度（平均、中央値）、追加トリプル数などの統計情報を計算し、ルール履歴に記録。
        # この評価結果は次回以降のルール選択とpool更新の判断材料となる。
        # 注: トリプルが追加されなかったルールは既にペナルティ記録済みなので、ここでは成功ルールのみ評価
        logger.info(f'Step {i}.4: Evaluating each rule and recording to history')
        logger.info(f'  [INPUT] Model before: {type(kge_current).__name__}')
        logger.info(f'  [INPUT] Model after: {type(kge_next).__name__}')
        logger.info(f'  [INPUT] Number of successful rules to evaluate: {len(successful_rules)}')
        analyzer = RuleWiseAnalyzer(kge_before=kge_current, kge_after=kge_next)
        
        for rule_with_id in successful_rules:
            rule_id = rule_with_id.rule_id
            rule_data = added_triples_by_rule[rule_id]
            
            logger.info(f'  Evaluating rule: {rule_id}')
            logger.info(f'    [INPUT] Target triples: {len(rule_data["target_triples"])}')
            logger.info(f'    [INPUT] Added triples: {len(rule_data["added_triples"])}')
            
            evaluation_record = analyzer.create_evaluation_record(
                iteration=i,
                rule_id=rule_id,
                rule=rule_data['rule'],
                target_triples=rule_data['target_triples'],
                added_triples=rule_data['added_triples']
            )
            
            rule_history.add_record(evaluation_record)
            logger.info(f'    [OUTPUT] Score change mean: {evaluation_record.mean_score_change:.6f}')
            logger.info(f'    [OUTPUT] Score change std: {evaluation_record.std_score_change:.6f}')
            logger.info(f'    [OUTPUT] Positive changes: {evaluation_record.positive_changes}')
            logger.info(f'    [OUTPUT] Negative changes: {evaluation_record.negative_changes}')
        
        # 履歴を保存
        history_pkl = os.path.join(dir_next, 'rule_history.pkl')
        history_json = os.path.join(dir_next, 'rule_history.json')
        rule_history.save(history_pkl)
        rule_history.save_json(history_json)
        logger.info(f'  [OUTPUT] Rule history saved to: {history_pkl}')
        logger.info(f'  [OUTPUT] Rule history (JSON) saved to: {history_json}')
        logger.info(f'  [OUTPUT] Total records in history: {len(rule_history.records)}')
        
        # 履歴サマリーレポート生成
        summary_report = rule_history.generate_summary_report()
        summary_file = os.path.join(dir_next, 'rule_history_summary.md')
        save_markdown(summary_report, summary_file)
        logger.info(f'  [OUTPUT] History summary report saved to: {summary_file}')
        
        # Step 5: Iteration全体の評価
        # 対象トリプルのスコア変化、追加トリプル数、知識グラフ埋め込み全体の精度（Hits@k, MRR）を
        # 評価し、iteration単位のレポートを生成。この評価結果は最終レポートに統合される。
        logger.info(f'Step {i}.5: Evaluating iteration performance')
        logger.info(f'  [INPUT] Evaluating {len(all_target_triples)} target triples')
        iteration_metrics = evaluator.evaluate_iteration(
            iteration=i,
            kge_before=kge_current,
            kge_after=kge_next,
            target_triples=all_target_triples,
            n_triples_added=len(all_added_triples),
            dir_save=dir_next
        )
        logger.info(f'  [OUTPUT] Iteration evaluation saved to: {dir_next}')
        
        # ルールpoolは固定（多腕バンディットの「腕」は変更しない）
        # 初期化時に作成したルールプールを最後まで使用する。
        # 各iterationでは選択のみを行い、ルール自体の追加・削除・変更は行わない。
        # 履歴情報は選択戦略（UCB, LLMポリシー等）の改善に使用される。
        logger.info(f'Step {i}.6: Rule pool remains fixed (multi-armed bandit arms)')
        logger.info(f'  [INFO] Rule pool size: {len(rule_pool)} (unchanged)')
        logger.info(f'  [INFO] Multi-armed bandit strategy: arms are not modified, only selection improves')
        
        # 現在のルールpoolを保存（参照用）
        pool_pkl = os.path.join(dir_next, 'rule_pool.pkl')
        pool_csv = os.path.join(dir_next, 'rule_pool.csv')
        rule_pool_amie.to_pickle(pool_pkl)
        rule_pool_amie.to_csv(pool_csv)
        
        logger.info(f'  [OUTPUT] Rule pool (copy) saved to: {pool_pkl}')
        logger.info(f'  [OUTPUT] Rule pool (CSV) saved to: {pool_csv}')
        
        # 次iterationの準備
        dir_current = dir_next
        kge_current = kge_next
        logger.info(f'  [STATE] Current directory updated to: {dir_current}')
        
        logger.info(f'Iteration {i} completed\n')
    
    logger.info('='*80)
    logger.info('All iterations completed!')
    logger.info('='*80)
    
    # 最終評価レポートの生成
    # 全iterationの評価結果をまとめ、追加トリプル数に対する対象トリプルのスコア改善、
    # 知識グラフ埋め込み全体の精度変化を可視化したグラフと共にMarkdownレポートを生成する。
    logger.info('Generating final evaluation report')
    evaluator.create_final_report(dir_save=dir_working)
    logger.info(f'Final evaluation report saved to: {dir_working}')
    
    # 最終レポート生成
    final_report = rule_history.generate_summary_report()
    save_markdown(final_report, os.path.join(dir_working, 'final_rule_history_summary.md'))
    
    logger.info(f'Results saved in: {dir_working}')
