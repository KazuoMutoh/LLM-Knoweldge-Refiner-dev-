"""
外部情報源から知識を取得し、レポートを生成するスクリプト

このスクリプトは以下の処理を行います:
1. FB15k-237データセットを読み込み、知識グラフを初期化
2. ルールファイル(rules.csv)を読み込み
3. ターゲットトリプルに対して外部情報を取得
4. 取得したエンティティに対して関連情報を取得
5. 取得したエンティティがKG内に同一のものがあるか調査
6. 結果をMarkdownレポートとして出力
"""

import sys
import os
from datetime import datetime
from typing import List, Dict, Set
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_active_refine.knoweldge_retriever import (
    TextAttributedKnoweldgeGraph, 
    LLMKnowledgeRetriever,
    KnowledgeRefiner,
    Entity,
    Relation,
    Triple,
    RetrievedKnowledge
)
from simple_active_refine.amie import AmieRules
from simple_active_refine.util import get_logger
from langchain_openai import ChatOpenAI

# ロガーの初期化
logger = get_logger('TestKnowledgeRetriever')

# =============================================================================
# 設定
# =============================================================================
DATA_DIR = './data/FB15k-237'
RULES_FILE = './playground/rules.csv'
OUTPUT_DIR = './playground/reports'
TARGET_TRIPLES = [
    ('/m/01wsl7c', '/people/person/nationality', '/m/07ssc')
]

# 出力ディレクトリの作成
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def main():
    """メイン処理"""
    logger.info("=" * 80)
    logger.info("外部情報取得スクリプトを開始します")
    logger.info("=" * 80)
    
    # ========================================================================
    # ステップ1: 知識グラフの初期化
    # ========================================================================
    logger.info("\n[ステップ1] 知識グラフの初期化")
    logger.info(f"データディレクトリ: {DATA_DIR}")
    kg = TextAttributedKnoweldgeGraph(dir_triples=DATA_DIR)
    logger.info(f"知識グラフの初期化が完了しました")
    logger.info(f"- train triples: {len(kg.triples.get('train', []))} 件")
    logger.info(f"- valid triples: {len(kg.triples.get('valid', []))} 件")
    logger.info(f"- test triples: {len(kg.triples.get('test', []))} 件")
    logger.info(f"- entities: {len(kg.entity_texts)} 件")
    logger.info(f"- relations: {len(kg.relation_texts)} 件")
    
    # ========================================================================
    # ステップ2: ルールの読み込み
    # ========================================================================
    logger.info("\n[ステップ2] ルールの読み込み")
    logger.info(f"ルールファイル: {RULES_FILE}")
    rules = AmieRules.from_csv(RULES_FILE)
    logger.info(f"ルールの読み込みが完了しました: {len(rules.rules)} 件")
    for i, rule in enumerate(rules.rules[:3], 1):
        logger.info(f"  ルール{i}: {rule}")
    
    # ========================================================================
    # ステップ3: ターゲットトリプルに対する外部情報取得
    # ========================================================================
    logger.info("\n[ステップ3] ターゲットトリプルに対する外部情報取得")
    logger.info(f"ターゲットトリプル数: {len(TARGET_TRIPLES)} 件")
    for i, triple in enumerate(TARGET_TRIPLES, 1):
        logger.info(f"  トリプル{i}: {triple[0]} -> {triple[1]} -> {triple[2]}")
    
    retriever = LLMKnowledgeRetriever(kg=kg)
    logger.info("LLMKnowledgeRetrieverを初期化しました")
    
    logger.info("外部情報の取得を開始します...")
    list_retrieved_knowledge: List[RetrievedKnowledge] = retriever.retrieve_knowledge(
        TARGET_TRIPLES, rules
    )
    logger.info(f"外部情報の取得が完了しました: {len(list_retrieved_knowledge)} 件")
    
    # 取得した情報のサマリーを出力
    total_triples = sum(len(rk.triples) for rk in list_retrieved_knowledge)
    total_entities = sum(len(rk.entities) for rk in list_retrieved_knowledge)
    logger.info(f"  - 取得したトリプル総数: {total_triples} 件")
    logger.info(f"  - 取得したエンティティ総数: {total_entities} 件")
    
    # 取得したエンティティの詳細をログ出力
    logger.info("\n取得したエンティティの詳細:")
    for rk_idx, rk in enumerate(list_retrieved_knowledge, 1):
        logger.info(f"  ルール {rk_idx} から取得:")
        for entity in rk.entities:
            logger.info(f"    - {entity.label} ({entity.id})")
            logger.info(f"      Short Description: {entity.description_short}")
            if entity.description:
                logger.info(f"      Description: {entity.description[:100]}..." if len(entity.description) > 100 else f"      Description: {entity.description}")
    
    # ステップ3のエンティティを保持
    entities_from_step3 = []
    for rk in list_retrieved_knowledge:
        entities_from_step3.extend(rk.entities)
    
    logger.info(f"ステップ3で取得したエンティティ数（重複含む）: {len(entities_from_step3)} 件")
    
    # スステップ3後のエンティティ重複排除（ステップ4の処理効率化のため）
    logger.info("\n[ステップ3後のエンティティ重複排除] ステップ4の処理を効率化するため")
    llm = ChatOpenAI(model='gpt-4o', temperature=0.1)
    entities_from_step3_unique, _ = deduplicate_entities_with_llm(entities_from_step3, llm, logger)
    logger.info(f"ステップ3後の重複排除完了: {len(entities_from_step3)} 件 -> {len(entities_from_step3_unique)} 件")
    
    # ========================================================================
    # ステップ4: 各エンティティに対する関連情報取得
    # ========================================================================
    logger.info("\n[ステップ4] 各エンティティに対する関連情報取得")
    
    # すべてのエンティティを収集（重複排除前のエンティティも保持しておく）
    all_entities_before_dedup: List[Entity] = entities_from_step3.copy()
    
    # ステップ4では重複排除後のユニークなエンティティのみを処理
    entities_for_step4 = entities_from_step3_unique
    
    logger.info(f"ステップ4の対象エンティティ数: {len(entities_for_step4)} 件（重複排除済み）")
    
    # 各エンティティに対して関連情報を取得
    # 注: 同じIDを持つエンティティが複数ある可能性があるため、インデックスをキーにする
    entity_related_knowledge: Dict[int, RetrievedKnowledge] = {}
    entity_index_mapping: Dict[int, Entity] = {}  # インデックス -> エンティティのマッピング
    
    for i, entity in enumerate(entities_for_step4):
        logger.info(f"  [{i+1}/{len(entities_for_step4)}] エンティティ '{entity.label}' ({entity.id}) の関連情報を取得中...")
        try:
            # 関連するリレーションのリストを取得（KGから全件）
            # relation_textsからRelationオブジェクトのリストを作成
            list_relations = [
                Relation(
                    id=rel_id,
                    label=rel_data.get('label', rel_id),
                    description_short=rel_data.get('description', ''),
                    description=rel_data.get('description', '')
                )
                for rel_id, rel_data in kg.relation_texts.items()
            ]
            
            related_knowledge = retriever.retrieve_knowledge_for_entity(
                entity, list_relations
            )
            entity_related_knowledge[i] = related_knowledge
            entity_index_mapping[i] = entity
            
            logger.info(f"    - 取得したトリプル数: {len(related_knowledge.triples)} 件")
            logger.info(f"    - 取得したエンティティ数: {len(related_knowledge.entities)} 件")
        except Exception as e:
            logger.error(f"    エラーが発生しました: {str(e)}")
            entity_related_knowledge[i] = RetrievedKnowledge(triples=[], entities=[])
            entity_index_mapping[i] = entity
    
    # 全エンティティを収集（重複排除前のステップ3エンティティ + ステップ4で取得した新エンティティ）
    all_entities_with_duplicates: List[Entity] = all_entities_before_dedup.copy()
    for idx, related_knowledge in entity_related_knowledge.items():
        for new_entity in related_knowledge.entities:
            all_entities_with_duplicates.append(new_entity)
    
    logger.info(f"関連情報取得後の総エンティティ数（重複含む）: {len(all_entities_with_duplicates)} 件")
    
    # ========================================================================
    # トリプルで使用されているエンティティのみを収集
    # ========================================================================
    logger.info("\n[トリプルで使用されているエンティティの収集]")
    
    # すべてのトリプルを収集
    all_triples: List[Triple] = []
    for rk in list_retrieved_knowledge:
        all_triples.extend(rk.triples)
    for idx, related_knowledge in entity_related_knowledge.items():
        all_triples.extend(related_knowledge.triples)
    
    logger.info(f"総トリプル数: {len(all_triples)} 件")
    
    # トリプルで使用されているエンティティIDを収集
    used_entity_ids = set()
    for triple in all_triples:
        used_entity_ids.add(triple.subject)
        used_entity_ids.add(triple.object)
    
    logger.info(f"トリプルで使用されているエンティティID数: {len(used_entity_ids)} 件")
    
    # トリプルで使用されているエンティティのみを全エンティティリストから抽出
    # 同じIDを持つエンティティが複数ある場合は全て保持（後で重複排除）
    entities_in_triples = [entity for entity in all_entities_with_duplicates if entity.id in used_entity_ids]
    
    logger.info(f"トリプルで使用されているエンティティ数（重複含む）: {len(entities_in_triples)} 件")
    
    # ========================================================================
    # エンティティの重複排除とID振り直し
    # ========================================================================
    logger.info("\n[最終エンティティ重複排除] トリプルで使用されているエンティティを対象にLLMで重複を判定し、ユニークなIDを振り直します")
    
    # 重複排除処理（トリプルで使用されているエンティティのみを対象）
    unique_entities, duplicate_groups = deduplicate_entities_with_llm(entities_in_triples, llm, logger)
    
    logger.info(f"重複排除後のエンティティ数: {len(unique_entities)} 件")
    
    # 重複グループ情報を使って、元のエンティティ -> ユニークエンティティのマッピングを作成
    logger.info("重複グループ情報からエンティティマッピングを作成しています...")
    entity_to_unique: Dict[int, Entity] = {}  # id(entity) -> unique_entity
    
    # まず、unique_entitiesの各エンティティを自分自身にマッピング
    for i, unique_entity in enumerate(unique_entities):
        # このユニークエンティティ自身をマッピング
        entity_to_unique[id(unique_entity)] = unique_entity
        
        # このユニークエンティティが重複グループの代表の場合、グループ内の全エンティティをマッピング
        for group in duplicate_groups:
            if i == group[0]:  # 最初のインデックスが代表
                # グループの全メンバー（代表も含む）を同じユニークエンティティにマッピング
                for idx in group:
                    entity_to_unique[id(entities_in_triples[idx])] = unique_entity
                break
    
    # トリプルのエンティティIDを更新する（ID振り直し前に実行）
    # 各トリプルのsubject/objectを、entity_to_uniqueマッピングを使って新しいエンティティに置き換える
    logger.info("トリプルのエンティティIDを更新しています...")
    logger.info(f"entity_to_unique マッピング数: {len(entity_to_unique)} 件")
    
    def find_entity_by_id(entity_id: str, entity_list: List[Entity]) -> Entity:
        """IDからエンティティを検索（最初に見つかったもの）"""
        for e in entity_list:
            if e.id == entity_id:
                return e
        return None
    
    updated_count = 0
    for rk in list_retrieved_knowledge:
        for triple in rk.triples:
            old_subject = triple.subject
            old_object = triple.object
            
            # subjectを更新
            subject_entity = find_entity_by_id(triple.subject, entities_in_triples)
            if subject_entity and id(subject_entity) in entity_to_unique:
                triple.subject = entity_to_unique[id(subject_entity)].id
                if old_subject != triple.subject:
                    updated_count += 1
                    logger.info(f"  トリプル subject 更新: {old_subject} -> {triple.subject}")
            
            # objectを更新
            object_entity = find_entity_by_id(triple.object, entities_in_triples)
            if object_entity and id(object_entity) in entity_to_unique:
                triple.object = entity_to_unique[id(object_entity)].id
                if old_object != triple.object:
                    updated_count += 1
                    logger.info(f"  トリプル object 更新: {old_object} -> {triple.object}")
    
    # entity_related_knowledgeのトリプルも更新
    for idx, related_knowledge in list(entity_related_knowledge.items()):
        for triple in related_knowledge.triples:
            old_subject = triple.subject
            old_object = triple.object
            
            # subjectを更新
            subject_entity = find_entity_by_id(triple.subject, entities_in_triples)
            if subject_entity and id(subject_entity) in entity_to_unique:
                triple.subject = entity_to_unique[id(subject_entity)].id
                if old_subject != triple.subject:
                    updated_count += 1
            
            # objectを更新
            object_entity = find_entity_by_id(triple.object, entities_in_triples)
            if object_entity and id(object_entity) in entity_to_unique:
                triple.object = entity_to_unique[id(object_entity)].id
                if old_object != triple.object:
                    updated_count += 1
    
    logger.info(f"トリプル更新完了: {updated_count} 件のエンティティ参照を更新")
    
    # ユニークなIDを振り直す（トリプル更新後に実行）
    # IDはlabelと同じにする（e1, e2, e3...）
    logger.info("ユニークなIDを振り直しています...")
    for i, entity in enumerate(unique_entities, 1):
        old_id = entity.id
        new_id = f"e{i}"
        entity.id = new_id
        entity.label = new_id  # labelもIDと同じにする
        logger.info(f"  エンティティ: {old_id} -> {new_id} (description: {entity.description_short})")
    
    # all_entities を unique_entities で置き換え
    all_entities = unique_entities
    logger.info(f"ID振り直し完了。最終エンティティ数: {len(all_entities)} 件")
    
    # entity_index_mappingを使って、最終的なentity_id -> RetrievedKnowledge のマッピングを作成
    final_entity_related_knowledge: Dict[str, RetrievedKnowledge] = {}
    for idx, related_knowledge in entity_related_knowledge.items():
        original_entity = entity_index_mapping[idx]
        # original_entityのIDが重複排除後に残っている場合、そのIDをマッピング
        for unique_entity in unique_entities:
            # description_shortで同一性を判定（labelは全てe1なので使えない）
            if (original_entity.description_short == unique_entity.description_short and
                original_entity.description == unique_entity.description):
                final_entity_related_knowledge[unique_entity.id] = related_knowledge
                break
    
    # ========================================================================
    # ステップ5: KG内の同一エンティティ検索
    # ========================================================================
    logger.info("\n[ステップ5] KG内の同一エンティティ検索")
    
    refiner = KnowledgeRefiner(kg=kg)
    logger.info("KnowledgeRefinerを初期化しました")
    
    # 各エンティティに対して同一エンティティを検索
    same_entity_mapping: Dict[str, List[Entity]] = {}
    
    for i, entity in enumerate(all_entities, 1):
        logger.info(f"  [{i}/{len(all_entities)}] エンティティ '{entity.label}' ({entity.id}) の同一エンティティを検索中...")
        try:
            same_entities = refiner.find_same_entity(
                entity, 
                top_k=5, 
                similarity_threshold=0.7
            )
            same_entity_mapping[entity.id] = same_entities
            
            if same_entities:
                logger.info(f"    - 同一エンティティ候補: {len(same_entities)} 件見つかりました")
                for se in same_entities:
                    logger.info(f"      - {se.label} ({se.id})")
            else:
                logger.info(f"    - 同一エンティティは見つかりませんでした")
        except Exception as e:
            logger.error(f"    エラーが発生しました: {str(e)}")
            same_entity_mapping[entity.id] = []
    
    # ========================================================================
    # ステップ6: Markdownレポート生成
    # ========================================================================
    logger.info("\n[ステップ6] Markdownレポート生成")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(OUTPUT_DIR, f'knowledge_retrieval_report_{timestamp}.md')
    
    logger.info(f"レポートを生成しています: {report_file}")
    
    generate_markdown_report(
        report_file=report_file,
        target_triples=TARGET_TRIPLES,
        list_retrieved_knowledge=list_retrieved_knowledge,
        entity_related_knowledge=final_entity_related_knowledge,
        all_entities=all_entities,
        same_entity_mapping=same_entity_mapping,
        kg=kg
    )
    
    logger.info(f"レポートの生成が完了しました: {report_file}")
    logger.info("\n" + "=" * 80)
    logger.info("すべての処理が完了しました")
    logger.info("=" * 80)


def check_duplicate_entities_with_llm(
    llm: ChatOpenAI,
    entity1: Entity,
    entity2: Entity,
    logger
) -> bool:
    """
    LLMを使用して2つのエンティティが重複（同一）かどうかを判定する
    
    Args:
        llm: ChatOpenAIのインスタンス
        entity1: 比較対象のエンティティ1
        entity2: 比較対象のエンティティ2
        logger: ロガー
    
    Returns:
        重複している場合はTrue、そうでない場合はFalse
    """
    # プロンプトの作成
    prompt = f"""以下の2つのエンティティが同一のものかどうか判定してください。

エンティティ1:
- Short Description: {entity1.description_short}
- Description: {entity1.description if entity1.description else "なし"}

エンティティ2:
- Short Description: {entity2.description_short}
- Description: {entity2.description if entity2.description else "なし"}

これらのエンティティが同一の実体を指している場合は "YES"、異なる実体を指している場合は "NO" とだけ回答してください。

重要な判断基準:
- Short Description（エンティティ名）とDescription（詳細説明）の内容**のみ**を使って判断すること
- Label（ラベル）は判断に使用しないこと（システム内部のIDなので無視する）
- 例: "St Andrews" と "United Kingdom" は明らかに異なる実体なので NO
- 例: "University of St Andrews" と "St Andrews University" は同じ実体の異なる表記なので YES
- 例: "Edinburgh"（都市）と "University of Edinburgh"（大学）は異なる実体なので NO

回答: """

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip().upper()
        
        is_duplicate = "YES" in answer
        
        if is_duplicate:
            logger.info(f"    重複判定: '{entity1.label}' と '{entity2.label}' は同一")
        
        return is_duplicate
    
    except Exception as e:
        logger.error(f"    LLMでの重複判定中にエラーが発生: {str(e)}")
        # エラー時は重複していないと判定
        return False


def deduplicate_entities_with_llm(
    entities: List[Entity],
    llm: ChatOpenAI,
    logger
) -> tuple[List[Entity], List[List[int]]]:
    """
    LLMを使用してエンティティリストの重複を排除する
    
    Args:
        entities: エンティティのリスト
        llm: ChatOpenAIのインスタンス
        logger: ロガー
    
    Returns:
        (重複を排除したエンティティのリスト, 重複グループのリスト)
        重複グループは、元のentitiesリスト内のインデックスのリスト
    """
    unique_entities: List[Entity] = []
    duplicate_groups: List[List[int]] = []
    processed_indices: Set[int] = set()
    
    logger.info(f"重複判定を開始します（対象エンティティ数: {len(entities)} 件）")
    
    for i, entity1 in enumerate(entities):
        if i in processed_indices:
            continue
        
        duplicate_indices = [i]
        
        for j in range(i + 1, len(entities)):
            if j in processed_indices:
                continue
            
            entity2 = entities[j]
            
            is_duplicate = check_duplicate_entities_with_llm(llm, entity1, entity2, logger)
            
            if is_duplicate:
                duplicate_indices.append(j)
                processed_indices.add(j)
        
        if len(duplicate_indices) > 1:
            duplicate_groups.append(duplicate_indices)
            logger.info(f"  重複グループを発見: {len(duplicate_indices)} 件のエンティティ")
            for idx in duplicate_indices:
                logger.info(f"    - {entities[idx].label} ({entities[idx].id})")
        
        unique_entities.append(entity1)
        processed_indices.add(i)
    
    logger.info(f"重複排除前: {len(entities)} 件")
    logger.info(f"重複排除後: {len(unique_entities)} 件")
    logger.info(f"削減されたエンティティ数: {len(entities) - len(unique_entities)} 件")
    logger.info(f"重複グループ数: {len(duplicate_groups)} グループ")
    
    return unique_entities, duplicate_groups


def generate_markdown_report(
    report_file: str,
    target_triples: List,
    list_retrieved_knowledge: List[RetrievedKnowledge],
    entity_related_knowledge: Dict[str, RetrievedKnowledge],
    all_entities: List[Entity],
    same_entity_mapping: Dict[str, List[Entity]],
    kg: TextAttributedKnoweldgeGraph
):
    """
    Markdownレポートを生成する
    
    Args:
        report_file: 出力ファイルパス
        target_triples: ターゲットトリプルのリスト
        list_retrieved_knowledge: ターゲットトリプルから取得した知識のリスト
        entity_related_knowledge: エンティティごとの関連知識
        all_entities: すべてのエンティティのリスト
        same_entity_mapping: エンティティIDから同一エンティティへのマッピング
        kg: 知識グラフ
    """
    with open(report_file, 'w', encoding='utf-8') as f:
        # ヘッダー
        f.write("# 外部情報取得レポート\n\n")
        f.write(f"**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # サマリー
        f.write("## サマリー\n\n")
        total_triples = sum(len(rk.triples) for rk in list_retrieved_knowledge)
        total_entities = len(all_entities)
        entities_with_same = sum(1 for entities in same_entity_mapping.values() if entities)
        
        f.write(f"- **ターゲットトリプル数**: {len(target_triples)}\n")
        f.write(f"- **取得したトリプル総数**: {total_triples}\n")
        f.write(f"- **取得したエンティティ総数**: {total_entities}\n")
        f.write(f"- **KG内に同一エンティティが見つかったもの**: {entities_with_same} / {total_entities}\n\n")
        f.write("---\n\n")
        
        # ターゲットトリプル
        f.write("## 1. ターゲットトリプル\n\n")
        for i, triple in enumerate(target_triples, 1):
            f.write(f"### トリプル {i}\n\n")
            
            # Subject名の取得（description_short優先）
            subject_name = triple[0]
            if triple[0] in kg.entity_texts:
                subj_data = kg.entity_texts[triple[0]]
                subject_name = subj_data.get('description_short', '') or subj_data.get('label', triple[0]) or triple[0]
            
            # Predicate名の簡略化
            predicate_name = triple[1].split('/')[-1].replace('_', ' ')
            
            # Object名の取得（description_short優先）
            object_name = triple[2]
            if triple[2] in kg.entity_texts:
                obj_data = kg.entity_texts[triple[2]]
                object_name = obj_data.get('description_short', '') or obj_data.get('label', triple[2]) or triple[2]
            
            f.write(f"- {subject_name} (`{triple[0]}`), {predicate_name}, {object_name} (`{triple[2]}`)\n\n")
            
            # 詳細情報を追加
            if triple[0] in kg.entity_texts:
                subj_data = kg.entity_texts[triple[0]]
                if subj_data.get('description'):
                    f.write(f"  - **Subject Description**: {subj_data.get('description')}\n")
            
            if triple[2] in kg.entity_texts:
                obj_data = kg.entity_texts[triple[2]]
                if obj_data.get('description'):
                    f.write(f"  - **Object Description**: {obj_data.get('description')}\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        
        # 取得したトリプル情報
        f.write("## 2. 取得したトリプル情報\n\n")
        
        all_triples = []
        for rk in list_retrieved_knowledge:
            all_triples.extend(rk.triples)
        
        # entity_related_knowledge からもトリプルを追加
        for entity_id, related_knowledge in entity_related_knowledge.items():
            all_triples.extend(related_knowledge.triples)
        
        if all_triples:
            f.write(f"取得したトリプル総数: **{len(all_triples)}** 件\n\n")
            
            # エンティティIDからインデックスへのマッピングを作成
            entity_id_to_index = {entity.id: idx + 1 for idx, entity in enumerate(all_entities)}
            
            for i, triple in enumerate(all_triples, 1):
                # Subject部分の生成
                if triple.subject in entity_id_to_index:
                    entity_idx = entity_id_to_index[triple.subject]
                    entity_obj = next((e for e in all_entities if e.id == triple.subject), None)
                    if entity_obj:
                        entity_name = entity_obj.description_short or entity_obj.label
                        subject_str = f"[{entity_name}](#エンティティ-{entity_idx}-{entity_obj.label.replace(' ', '-').lower()})"
                    else:
                        subject_str = f"`{triple.subject}`"
                elif triple.subject in kg.entity_texts:
                    subj_data = kg.entity_texts[triple.subject]
                    subject_str = subj_data.get('description_short', '') or subj_data.get('label', triple.subject) or triple.subject
                else:
                    subject_str = f"`{triple.subject}`"
                
                # Predicate部分の生成（関係名を簡略化）
                predicate_name = triple.predicate.split('/')[-1].replace('_', ' ')
                
                # Object部分の生成
                if triple.object in entity_id_to_index:
                    entity_idx = entity_id_to_index[triple.object]
                    entity_obj = next((e for e in all_entities if e.id == triple.object), None)
                    if entity_obj:
                        entity_name = entity_obj.description_short or entity_obj.label
                        object_str = f"[{entity_name}](#エンティティ-{entity_idx}-{entity_obj.label.replace(' ', '-').lower()}) (`{triple.object}`)"
                    else:
                        object_str = f"`{triple.object}`"
                elif triple.object in kg.entity_texts:
                    obj_data = kg.entity_texts[triple.object]
                    object_str = obj_data.get('description_short', '') or obj_data.get('label', triple.object) or triple.object
                else:
                    object_str = f"`{triple.object}`"
                
                # Source部分の生成
                source_str = f" ([source]({triple.source}))" if triple.source else ""
                
                # 1行で表示（見出しなし）
                f.write(f"{i}. {subject_str}, {predicate_name}, {object_str}{source_str}\n")
        else:
            f.write("取得したトリプルはありません。\n\n")
        
        f.write("---\n\n")
        
        # 取得したエンティティ情報
        f.write("## 3. 取得したエンティティ情報\n\n")
        f.write(f"取得したエンティティ総数: **{len(all_entities)}** 件\n\n")
        
        # すべてのトリプルを取得（エンティティとトリプルの関連付けのため）
        all_triples = []
        for rk in list_retrieved_knowledge:
            all_triples.extend(rk.triples)
        for entity_id, related_knowledge in entity_related_knowledge.items():
            all_triples.extend(related_knowledge.triples)
        
        for i, entity in enumerate(all_entities, 1):
            f.write(f"### エンティティ {i}: {entity.label}\n\n")
            
            # 基本情報
            f.write("#### 基本情報\n\n")
            f.write(f"- **ID**: `{entity.id}`\n")
            f.write(f"- **Label**: {entity.label}\n")
            f.write(f"- **Short Description**: {entity.description_short}\n")
            
            if entity.description:
                f.write(f"- **Description**: {entity.description}\n")
            
            if entity.source:
                f.write(f"- **Source**: {entity.source}\n")
            
            f.write("\n")
            
            # このエンティティが含まれるトリプル
            f.write("#### このエンティティが含まれるトリプル\n\n")
            
            # エンティティIDからインデックスへのマッピング
            entity_id_to_index = {ent.id: idx + 1 for idx, ent in enumerate(all_entities)}
            
            # このエンティティがSubjectまたはObjectとして含まれるトリプルを検索
            related_triples = [t for t in all_triples if t.subject == entity.id or t.object == entity.id]
            
            if related_triples:
                f.write(f"このエンティティが含まれるトリプル数: **{len(related_triples)}** 件\n\n")
                
                for j, triple in enumerate(related_triples, 1):
                    # Subject部分の生成
                    if triple.subject == entity.id:
                        subject_str = f"{entity.description_short or entity.label} (このエンティティ)"
                    elif triple.subject in entity_id_to_index:
                        subj_idx = entity_id_to_index[triple.subject]
                        subj_obj = next((e for e in all_entities if e.id == triple.subject), None)
                        if subj_obj:
                            subj_name = subj_obj.description_short or subj_obj.label
                            subject_str = f"[{subj_name}](#エンティティ-{subj_idx}-{subj_obj.label.replace(' ', '-').lower()})"
                        else:
                            subject_str = f"`{triple.subject}`"
                    elif triple.subject in kg.entity_texts:
                        subj_data = kg.entity_texts[triple.subject]
                        subject_str = subj_data.get('description_short', '') or subj_data.get('label', triple.subject) or triple.subject
                    else:
                        subject_str = f"`{triple.subject}`"
                    
                    # Predicate部分の生成（関係名を簡略化）
                    predicate_name = triple.predicate.split('/')[-1].replace('_', ' ')
                    
                    # Object部分の生成
                    if triple.object == entity.id:
                        object_str = f"{entity.description_short or entity.label} (このエンティティ)"
                    elif triple.object in entity_id_to_index:
                        obj_idx = entity_id_to_index[triple.object]
                        obj_obj = next((e for e in all_entities if e.id == triple.object), None)
                        if obj_obj:
                            obj_name = obj_obj.description_short or obj_obj.label
                            object_str = f"[{obj_name}](#エンティティ-{obj_idx}-{obj_obj.label.replace(' ', '-').lower()}) (`{triple.object}`)"
                        else:
                            object_str = f"`{triple.object}`"
                    elif triple.object in kg.entity_texts:
                        obj_data = kg.entity_texts[triple.object]
                        object_str = obj_data.get('description_short', '') or obj_data.get('label', triple.object) or triple.object
                    else:
                        object_str = f"`{triple.object}`"
                    
                    # Source部分の生成
                    source_str = f" ([source]({triple.source}))" if triple.source else ""
                    
                    # 1行で表示（見出しなし、番号付きリスト）
                    f.write(f"{j}. {subject_str}, {predicate_name}, {object_str}{source_str}\n")
            else:
                f.write("このエンティティが含まれるトリプルはありません。\n\n")
            
            # KG内の同一エンティティ
            f.write("#### KG内の同一エンティティ\n\n")
            
            same_entities = same_entity_mapping.get(entity.id, [])
            
            if same_entities:
                f.write(f"**同一エンティティが見つかりました**: {len(same_entities)} 件\n\n")
                
                for j, same_entity in enumerate(same_entities, 1):
                    f.write(f"##### 同一エンティティ {j}\n\n")
                    f.write(f"- **ID**: `{same_entity.id}`\n")
                    f.write(f"- **Label**: {same_entity.label}\n")
                    f.write(f"- **Short Description**: {same_entity.description_short}\n")
                    
                    if same_entity.description:
                        f.write(f"- **Description**: {same_entity.description}\n")
                    
                    f.write("\n")
            else:
                f.write("**同一エンティティは見つかりませんでした。**\n\n")
            
            # このエンティティに関連する取得情報
            if entity.id in entity_related_knowledge:
                related = entity_related_knowledge[entity.id]
                if related.triples or related.entities:
                    f.write("#### このエンティティから取得した関連情報\n\n")
                    
                    if related.triples:
                        f.write(f"- **関連トリプル数**: {len(related.triples)}\n")
                    
                    if related.entities:
                        f.write(f"- **関連エンティティ数**: {len(related.entities)}\n")
                    
                    f.write("\n")
            
            f.write("---\n\n")
        
        # フッター
        f.write("## レポート生成完了\n\n")
        f.write("このレポートは自動生成されました。\n")


if __name__ == '__main__':
    main()
