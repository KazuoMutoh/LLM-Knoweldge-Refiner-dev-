"""
retrieve_knowledge_for_entityメソッドのデモンストレーション

このスクリプトは、LLMKnowledgeRetrieverクラスの新しいメソッド
retrieve_knowledge_for_entityの使用方法を示します。
"""
import os
import sys
from simple_active_refine.knoweldge_retriever import (
    TextAttributedKnoweldgeGraph,
    LLMKnowledgeRetriever,
    Entity,
    Relation
)

def main():
    print("=" * 80)
    print("retrieve_knowledge_for_entity メソッドのデモンストレーション")
    print("=" * 80)
    
    # 1. 知識グラフの初期化
    print("\n1. 知識グラフを初期化中...")
    data_dir = "/app/data/FB15k-237"  # 既存のデータセットを使用
    kg = TextAttributedKnoweldgeGraph(data_dir)
    print(f"   ✓ 知識グラフの初期化完了（エンティティ数: {len(kg.entity_texts)}）")
    
    # 2. LLMKnowledgeRetrieverの初期化
    print("\n2. LLMKnowledgeRetrieverを初期化中...")
    retriever = LLMKnowledgeRetriever(kg=kg, llm_model='gpt-4o', use_web_search=True)
    print("   ✓ LLMKnowledgeRetrieverの初期化完了")
    
    # 3. テスト用のエンティティを作成
    print("\n3. テスト用のエンティティを作成...")
    test_entity = Entity(
        id="/m/test_elon_musk",
        label="/m/test_elon_musk",
        description_short="Elon Musk",
        description="Elon Musk is a business magnate and entrepreneur. He is the founder, CEO, and chief engineer of SpaceX; early-stage investor, CEO, and product architect of Tesla, Inc."
    )
    print(f"   エンティティ: {test_entity.label}")
    print(f"   説明: {test_entity.description_short}")
    
    # 4. 候補となるRelationのリストを作成
    print("\n4. 候補となるRelationのリストを作成...")
    candidate_relations = [
        Relation(
            id="born_in",
            label="born_in",
            description_short="was born in",
            description="Indicates the place where a person was born"
        ),
        Relation(
            id="nationality",
            label="nationality",
            description_short="has nationality",
            description="Indicates the country of citizenship or nationality of a person"
        ),
        Relation(
            id="founded",
            label="founded",
            description_short="founded",
            description="Indicates an organization or company founded by a person"
        ),
        Relation(
            id="occupation",
            label="occupation",
            description_short="has occupation",
            description="Indicates the profession or job of a person"
        ),
        Relation(
            id="award",
            label="award",
            description_short="received award",
            description="Indicates an award or honor received by a person"
        )
    ]
    print(f"   候補Relation数: {len(candidate_relations)}")
    for rel in candidate_relations:
        print(f"     - {rel.id}: {rel.description_short}")
    
    # 5. retrieve_knowledge_for_entityメソッドを実行
    print("\n5. retrieve_knowledge_for_entityメソッドを実行中...")
    print("   (LLMがrelationを選択し、webから情報を取得します...)")
    
    retrieved_knowledge = retriever.retrieve_knowledge_for_entity(
        entity=test_entity,
        list_relations=candidate_relations
    )
    
    # 6. 結果を表示
    print("\n" + "=" * 80)
    print("取得結果")
    print("=" * 80)
    
    print(f"\n取得されたトリプル数: {len(retrieved_knowledge.triples)}")
    if retrieved_knowledge.triples:
        print("\nトリプル:")
        for i, triple in enumerate(retrieved_knowledge.triples, 1):
            print(f"  {i}. ({triple.subject}, {triple.predicate}, {triple.object})")
            if triple.source:
                print(f"     ソース: {triple.source}")
    
    print(f"\n取得されたエンティティ数: {len(retrieved_knowledge.entities)}")
    if retrieved_knowledge.entities:
        print("\nエンティティ:")
        for i, entity in enumerate(retrieved_knowledge.entities, 1):
            print(f"  {i}. ID: {entity.id}")
            print(f"     Label: {entity.label}")
            print(f"     説明（短）: {entity.description_short}")
            if entity.description:
                print(f"     説明（長）: {entity.description[:100]}...")
            if entity.source:
                print(f"     ソース: {entity.source}")
            print()
    
    # 7. 知識グラフに追加（オプション）
    print("\n7. 取得した知識を知識グラフに追加...")
    if retrieved_knowledge.triples or retrieved_knowledge.entities:
        # 注: 実際の運用では、取得した知識を検証してから追加することを推奨
        # kg.add_retrieved_knowledge(retrieved_knowledge, data_type='train')
        print("   (デモのため、実際の追加はスキップします)")
    
    print("\n" + "=" * 80)
    print("デモンストレーション完了")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
