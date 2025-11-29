"""
KnowledgeRefinerクラスのテストコード

テスト内容:
1. KnowledgeRefinerの初期化
2. find_same_entityメソッドで同一エンティティを正しく検出できるか
3. find_same_entityメソッドで異なるエンティティを正しく区別できるか
"""
import os
import sys
import tempfile
import shutil
import pytest
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simple_active_refine.knoweldge_retriever import (
    TextAttributedKnoweldgeGraph,
    KnowledgeRefiner,
    Entity
)


@pytest.fixture
def temp_kg_dir():
    """テスト用の一時ディレクトリとテストデータを作成"""
    temp_dir = tempfile.mkdtemp()
    
    # テスト用のトリプルデータを作成
    train_data = [
        "/m/obama\tborn_in\t/m/hawaii",
        "/m/hawaii\tlocated_in\t/m/usa",
        "/m/usa\tcapital\t/m/washington"
    ]
    
    # entity2text.txtを作成
    entity_data = [
        "/m/obama\tBarack Obama",
        "/m/hawaii\tHawaii",
        "/m/usa\tUnited States of America",
        "/m/washington\tWashington D.C."
    ]
    
    # entity2textlong.txtを作成
    entity_long_data = [
        "/m/obama\tBarack Hussein Obama II is an American politician who served as the 44th president of the United States from 2009 to 2017.",
        "/m/hawaii\tHawaii is a state in the Western United States, located in the Pacific Ocean.",
        "/m/usa\tThe United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or America, is a country primarily located in North America.",
        "/m/washington\tWashington, D.C., formally the District of Columbia, also known as just Washington or simply D.C., is the capital city and federal district of the United States."
    ]
    
    # relation2text.txtを作成
    relation_data = [
        "born_in\twas born in",
        "located_in\tis located in",
        "capital\thas capital"
    ]
    
    # ファイルを書き込み
    with open(os.path.join(temp_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_data))
    
    with open(os.path.join(temp_dir, 'entity2text.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(entity_data))
    
    with open(os.path.join(temp_dir, 'entity2textlong.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(entity_long_data))
    
    with open(os.path.join(temp_dir, 'relation2text.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(relation_data))
    
    yield temp_dir
    
    # テスト後にクリーンアップ
    shutil.rmtree(temp_dir)


@pytest.fixture
def kg(temp_kg_dir):
    """テスト用のTextAttributedKnoweldgeGraphインスタンスを作成"""
    return TextAttributedKnoweldgeGraph(temp_kg_dir)


@pytest.fixture
def refiner(kg):
    """テスト用のKnowledgeRefinerインスタンスを作成"""
    return KnowledgeRefiner(kg)


def test_knowledge_refiner_initialization(refiner):
    """KnowledgeRefinerが正しく初期化されるかテスト"""
    assert refiner is not None
    assert refiner.kg is not None
    assert refiner.llm is not None
    print("✓ KnowledgeRefiner初期化テスト: 成功")


def test_find_same_entity_with_similar_entity(refiner, kg):
    """
    類似したエンティティが同一と判断されるかテスト
    
    新しいエンティティ（Barack Obama）を作成し、既存のエンティティ（/m/obama）と
    同一かどうかを判断させる
    """
    # テスト用の新しいエンティティを作成（既存の/m/obamaと同じ人物）
    new_entity = Entity(
        id="e_new_obama",
        label="e_new_obama",
        description_short="Barack Obama",
        description="Barack Obama is an American politician who was the 44th president of the United States."
    )
    
    # 既存KGに追加して検索可能にする
    kg.add_entities([new_entity])
    
    # 既存エンティティを取得
    existing_entity = Entity(
        id="/m/obama",
        label="/m/obama",
        description_short="Barack Obama",
        description="Barack Hussein Obama II is an American politician who served as the 44th president of the United States from 2009 to 2017."
    )
    
    # find_same_entityを実行
    same_entities = refiner.find_same_entity(existing_entity, top_k=5, similarity_threshold=0.5)
    
    # 結果を検証
    print(f"\n検索結果: {len(same_entities)}件のマッチ")
    for entity in same_entities:
        print(f"  - {entity.id}: {entity.description_short}")
    
    # 新しいエンティティが同一と判断されることを確認
    assert len(same_entities) > 0, "同一エンティティが見つかりませんでした"
    assert any(e.id == "e_new_obama" for e in same_entities), "新しいエンティティが同一と判断されませんでした"
    
    print("✓ 類似エンティティの同一性検出テスト: 成功")


def test_find_same_entity_with_different_entity(refiner, kg):
    """
    異なるエンティティが正しく区別されるかテスト
    
    全く異なるエンティティを作成し、既存のエンティティと異なると判断されるかテスト
    """
    # 全く異なるエンティティを作成
    different_entity = Entity(
        id="e_different",
        label="e_different",
        description_short="Albert Einstein",
        description="Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest physicists of all time."
    )
    
    # 既存KGに追加
    kg.add_entities([different_entity])
    
    # 既存エンティティ（Barack Obama）を取得
    existing_entity = Entity(
        id="/m/obama",
        label="/m/obama",
        description_short="Barack Obama",
        description="Barack Hussein Obama II is an American politician who served as the 44th president of the United States from 2009 to 2017."
    )
    
    # find_same_entityを実行
    same_entities = refiner.find_same_entity(existing_entity, top_k=5, similarity_threshold=0.3)
    
    # 結果を検証
    print(f"\n検索結果: {len(same_entities)}件のマッチ")
    for entity in same_entities:
        print(f"  - {entity.id}: {entity.description_short}")
    
    # 異なるエンティティ（Albert Einstein）が同一と判断されないことを確認
    einstein_ids = [e.id for e in same_entities if e.id == "e_different"]
    assert len(einstein_ids) == 0, "異なるエンティティ（Einstein）が誤って同一と判断されました"
    
    print("✓ 異なるエンティティの区別テスト: 成功")


def test_find_same_entity_no_matches(refiner):
    """
    マッチするエンティティがない場合に空のリストが返されるかテスト
    """
    # 全く新しいエンティティ（KGに存在しない）
    new_entity = Entity(
        id="e_completely_new",
        label="e_completely_new",
        description_short="Marie Curie",
        description="Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity."
    )
    
    # find_same_entityを実行（このエンティティは追加していないので類似エンティティも見つからない）
    same_entities = refiner.find_same_entity(new_entity, top_k=5, similarity_threshold=0.7)
    
    # 結果を検証
    print(f"\n検索結果: {len(same_entities)}件のマッチ")
    assert isinstance(same_entities, list), "戻り値がリストではありません"
    print("✓ マッチなし時の空リスト返却テスト: 成功")


if __name__ == "__main__":
    # pytestを使わずに直接実行する場合
    print("=" * 70)
    print("KnowledgeRefinerテストを開始します")
    print("=" * 70)
    
    # 一時ディレクトリとKGを作成
    temp_dir = tempfile.mkdtemp()
    
    try:
        # テストデータを作成
        train_data = [
            "/m/obama\tborn_in\t/m/hawaii",
            "/m/hawaii\tlocated_in\t/m/usa",
            "/m/usa\tcapital\t/m/washington"
        ]
        
        entity_data = [
            "/m/obama\tBarack Obama",
            "/m/hawaii\tHawaii",
            "/m/usa\tUnited States of America",
            "/m/washington\tWashington D.C."
        ]
        
        entity_long_data = [
            "/m/obama\tBarack Hussein Obama II is an American politician who served as the 44th president of the United States from 2009 to 2017.",
            "/m/hawaii\tHawaii is a state in the Western United States, located in the Pacific Ocean.",
            "/m/usa\tThe United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or America, is a country primarily located in North America.",
            "/m/washington\tWashington, D.C., formally the District of Columbia, also known as just Washington or simply D.C., is the capital city and federal district of the United States."
        ]
        
        relation_data = [
            "born_in\twas born in",
            "located_in\tis located in",
            "capital\thas capital"
        ]
        
        with open(os.path.join(temp_dir, 'train.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_data))
        
        with open(os.path.join(temp_dir, 'entity2text.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(entity_data))
        
        with open(os.path.join(temp_dir, 'entity2textlong.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(entity_long_data))
        
        with open(os.path.join(temp_dir, 'relation2text.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(relation_data))
        
        # KGとRefinerを作成
        print("\n1. 知識グラフを初期化中...")
        kg = TextAttributedKnoweldgeGraph(temp_dir)
        print("   知識グラフの初期化完了")
        
        print("\n2. KnowledgeRefinerを初期化中...")
        refiner = KnowledgeRefiner(kg)
        print("   KnowledgeRefinerの初期化完了")
        
        # テスト1: 初期化テスト
        print("\n" + "=" * 70)
        print("テスト1: 初期化テスト")
        print("=" * 70)
        test_knowledge_refiner_initialization(refiner)
        
        # テスト2: 類似エンティティの検出
        print("\n" + "=" * 70)
        print("テスト2: 類似エンティティの同一性検出テスト")
        print("=" * 70)
        test_find_same_entity_with_similar_entity(refiner, kg)
        
        # テスト3: 異なるエンティティの区別
        print("\n" + "=" * 70)
        print("テスト3: 異なるエンティティの区別テスト")
        print("=" * 70)
        test_find_same_entity_with_different_entity(refiner, kg)
        
        # テスト4: マッチなしの場合
        print("\n" + "=" * 70)
        print("テスト4: マッチなし時の空リスト返却テスト")
        print("=" * 70)
        test_find_same_entity_no_matches(refiner)
        
        print("\n" + "=" * 70)
        print("全てのテストが成功しました！")
        print("=" * 70)
        
    finally:
        # クリーンアップ
        shutil.rmtree(temp_dir)
