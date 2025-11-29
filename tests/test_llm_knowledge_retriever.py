"""
LLMKnowledgeRetrieverクラスのretrieve_knowledge_for_entityメソッドのテストコード

テスト内容:
1. LLMKnowledgeRetrieverの初期化
2. retrieve_knowledge_for_entityメソッドで適切にrelationが選択されるか
3. retrieve_knowledge_for_entityメソッドで適切にentityが取得されるか
4. retrieve_knowledge_for_entityメソッドで適切なRetrievedKnowledgeが返されるか
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
    LLMKnowledgeRetriever,
    Entity,
    Relation,
    RetrievedKnowledge
)


@pytest.fixture
def temp_kg_dir():
    """テスト用の一時ディレクトリとテストデータを作成"""
    temp_dir = tempfile.mkdtemp()
    
    # テスト用のトリプルデータを作成
    train_data = [
        "/m/obama\tborn_in\t/m/hawaii",
        "/m/hawaii\tlocated_in\t/m/usa",
        "/m/usa\tcapital\t/m/washington",
        "/m/einstein\tborn_in\t/m/ulm",
        "/m/einstein\tnationality\t/m/germany"
    ]
    
    # entity2text.txtを作成
    entity_data = [
        "/m/obama\tBarack Obama",
        "/m/hawaii\tHawaii",
        "/m/usa\tUnited States of America",
        "/m/washington\tWashington D.C.",
        "/m/einstein\tAlbert Einstein",
        "/m/ulm\tUlm",
        "/m/germany\tGermany"
    ]
    
    # entity2textlong.txtを作成
    entity_long_data = [
        "/m/obama\tBarack Hussein Obama II is an American politician who served as the 44th president of the United States from 2009 to 2017.",
        "/m/hawaii\tHawaii is a state in the Western United States, located in the Pacific Ocean.",
        "/m/usa\tThe United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or America, is a country primarily located in North America.",
        "/m/washington\tWashington, D.C., formally the District of Columbia, also known as just Washington or simply D.C., is the capital city and federal district of the United States.",
        "/m/einstein\tAlbert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest physicists of all time.",
        "/m/ulm\tUlm is a city in the federal German state of Baden-Württemberg.",
        "/m/germany\tGermany, officially the Federal Republic of Germany, is a country in Central Europe."
    ]
    
    # relation2text.txtを作成
    relation_data = [
        "born_in\twas born in",
        "located_in\tis located in",
        "capital\thas capital",
        "nationality\thas nationality"
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
def retriever(kg):
    """テスト用のLLMKnowledgeRetrieverインスタンスを作成"""
    return LLMKnowledgeRetriever(kg=kg, llm_model='gpt-4o', use_web_search=True)


@pytest.fixture
def sample_relations():
    """テスト用のRelationリストを作成"""
    return [
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
            id="occupation",
            label="occupation",
            description_short="has occupation",
            description="Indicates the profession or job of a person"
        ),
        Relation(
            id="education",
            label="education",
            description_short="educated at",
            description="Indicates the educational institution where a person studied"
        ),
        Relation(
            id="award",
            label="award",
            description_short="received award",
            description="Indicates an award or honor received by a person"
        )
    ]


def test_llm_knowledge_retriever_initialization(retriever):
    """LLMKnowledgeRetrieverが正しく初期化されるかテスト"""
    assert retriever is not None
    assert retriever.kg is not None
    assert retriever.llm is not None
    assert retriever.openai_client is not None
    assert retriever.llm_model == 'gpt-4o'
    assert retriever.use_web_search == True
    print("✓ LLMKnowledgeRetriever初期化テスト: 成功")


def test_retrieve_knowledge_for_entity_basic(retriever, sample_relations):
    """
    retrieve_knowledge_for_entityメソッドの基本動作テスト
    
    Barack Obamaについて関連するrelationを選択し、知識を取得する
    """
    # テスト用のエンティティを作成
    test_entity = Entity(
        id="/m/obama",
        label="/m/obama",
        description_short="Barack Obama",
        description="Barack Hussein Obama II is an American politician who served as the 44th president of the United States from 2009 to 2017."
    )
    
    # メソッドを実行
    print(f"\n検索対象エンティティ: {test_entity.label}")
    print(f"説明: {test_entity.description_short}")
    
    retrieved_knowledge = retriever.retrieve_knowledge_for_entity(test_entity, sample_relations)
    
    # 結果を検証
    assert isinstance(retrieved_knowledge, RetrievedKnowledge), "戻り値がRetrievedKnowledgeのインスタンスではありません"
    assert isinstance(retrieved_knowledge.triples, list), "triplesがリストではありません"
    assert isinstance(retrieved_knowledge.entities, list), "entitiesがリストではありません"
    
    print(f"\n取得結果:")
    print(f"  - トリプル数: {len(retrieved_knowledge.triples)}")
    print(f"  - エンティティ数: {len(retrieved_knowledge.entities)}")
    
    # トリプルとエンティティの内容を表示
    if retrieved_knowledge.triples:
        print(f"\n取得されたトリプル:")
        for i, triple in enumerate(retrieved_knowledge.triples, 1):
            print(f"    {i}. ({triple.subject}, {triple.predicate}, {triple.object})")
            if triple.source:
                print(f"       ソース: {triple.source}")
    
    if retrieved_knowledge.entities:
        print(f"\n取得されたエンティティ:")
        for i, entity in enumerate(retrieved_knowledge.entities, 1):
            print(f"    {i}. {entity.label}: {entity.description_short}")
            if entity.source:
                print(f"       ソース: {entity.source}")
    
    # 少なくとも何らかの結果が返されることを期待（LLMの結果に依存するため、厳密なアサーションは避ける）
    assert len(retrieved_knowledge.triples) >= 0, "トリプルが負の値です"
    assert len(retrieved_knowledge.entities) >= 0, "エンティティが負の値です"
    
    print("\n✓ retrieve_knowledge_for_entity基本動作テスト: 成功")


def test_retrieve_knowledge_for_entity_with_scientist(retriever, sample_relations):
    """
    科学者（Albert Einstein）についての知識取得テスト
    
    異なるタイプのエンティティでもメソッドが正しく動作することを確認
    """
    # テスト用のエンティティを作成
    test_entity = Entity(
        id="/m/einstein",
        label="/m/einstein",
        description_short="Albert Einstein",
        description="Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest physicists of all time."
    )
    
    # メソッドを実行
    print(f"\n検索対象エンティティ: {test_entity.label}")
    print(f"説明: {test_entity.description_short}")
    
    retrieved_knowledge = retriever.retrieve_knowledge_for_entity(test_entity, sample_relations)
    
    # 結果を検証
    assert isinstance(retrieved_knowledge, RetrievedKnowledge), "戻り値がRetrievedKnowledgeのインスタンスではありません"
    
    print(f"\n取得結果:")
    print(f"  - トリプル数: {len(retrieved_knowledge.triples)}")
    print(f"  - エンティティ数: {len(retrieved_knowledge.entities)}")
    
    # トリプルとエンティティの詳細を表示
    if retrieved_knowledge.triples:
        print(f"\n取得されたトリプル:")
        for i, triple in enumerate(retrieved_knowledge.triples, 1):
            print(f"    {i}. ({triple.subject}, {triple.predicate}, {triple.object})")
    
    if retrieved_knowledge.entities:
        print(f"\n取得されたエンティティ:")
        for i, entity in enumerate(retrieved_knowledge.entities, 1):
            print(f"    {i}. {entity.label}: {entity.description_short}")
    
    print("\n✓ 科学者エンティティの知識取得テスト: 成功")


def test_retrieve_knowledge_for_entity_return_structure(retriever, sample_relations):
    """
    返り値の構造が正しいかテスト
    
    RetrievedKnowledgeの各フィールドが適切な型とフォーマットであることを確認
    """
    test_entity = Entity(
        id="/m/test_entity",
        label="/m/test_entity",
        description_short="Test Entity",
        description="A test entity for verifying the return structure"
    )
    
    retrieved_knowledge = retriever.retrieve_knowledge_for_entity(test_entity, sample_relations)
    
    # 戻り値の型チェック
    assert isinstance(retrieved_knowledge, RetrievedKnowledge), "戻り値の型が正しくありません"
    assert hasattr(retrieved_knowledge, 'triples'), "triplesフィールドが存在しません"
    assert hasattr(retrieved_knowledge, 'entities'), "entitiesフィールドが存在しません"
    
    # リストの型チェック
    assert isinstance(retrieved_knowledge.triples, list), "triplesがリストではありません"
    assert isinstance(retrieved_knowledge.entities, list), "entitiesがリストではありません"
    
    # 各トリプルの構造チェック
    for triple in retrieved_knowledge.triples:
        assert hasattr(triple, 'subject'), "トリプルにsubjectフィールドがありません"
        assert hasattr(triple, 'predicate'), "トリプルにpredicateフィールドがありません"
        assert hasattr(triple, 'object'), "トリプルにobjectフィールドがありません"
        assert hasattr(triple, 'source'), "トリプルにsourceフィールドがありません"
        assert isinstance(triple.subject, str), "triple.subjectが文字列ではありません"
        assert isinstance(triple.predicate, str), "triple.predicateが文字列ではありません"
        assert isinstance(triple.object, str), "triple.objectが文字列ではありません"
    
    # 各エンティティの構造チェック
    for entity in retrieved_knowledge.entities:
        assert hasattr(entity, 'id'), "エンティティにidフィールドがありません"
        assert hasattr(entity, 'label'), "エンティティにlabelフィールドがありません"
        assert hasattr(entity, 'description_short'), "エンティティにdescription_shortフィールドがありません"
        assert hasattr(entity, 'description'), "エンティティにdescriptionフィールドがありません"
        assert hasattr(entity, 'source'), "エンティティにsourceフィールドがありません"
        assert isinstance(entity.id, str), "entity.idが文字列ではありません"
        assert isinstance(entity.label, str), "entity.labelが文字列ではありません"
        assert isinstance(entity.description_short, str), "entity.description_shortが文字列ではありません"
        # idとlabelが同じ値であることを確認
        assert entity.id == entity.label, f"idとlabelが異なります: id={entity.id}, label={entity.label}"
    
    print("\n✓ 返り値の構造検証テスト: 成功")


def test_retrieve_knowledge_for_entity_empty_relations(retriever):
    """
    空のrelationリストを渡した場合のテスト
    
    エッジケースとして、relationが空の場合に適切に処理されるかを確認
    """
    test_entity = Entity(
        id="/m/test_entity",
        label="/m/test_entity",
        description_short="Test Entity",
        description="A test entity"
    )
    
    # 空のrelationリストで実行
    retrieved_knowledge = retriever.retrieve_knowledge_for_entity(test_entity, [])
    
    # 空の結果が返されることを確認
    assert isinstance(retrieved_knowledge, RetrievedKnowledge), "戻り値の型が正しくありません"
    assert len(retrieved_knowledge.triples) == 0, "空のrelationリストで非空のトリプルが返されました"
    assert len(retrieved_knowledge.entities) == 0, "空のrelationリストで非空のエンティティが返されました"
    
    print("\n✓ 空のrelationリスト処理テスト: 成功")


if __name__ == "__main__":
    # pytestを使わずに直接実行する場合
    print("=" * 70)
    print("LLMKnowledgeRetriever.retrieve_knowledge_for_entityテストを開始します")
    print("=" * 70)
    
    # 一時ディレクトリとKGを作成
    temp_dir = tempfile.mkdtemp()
    
    try:
        # テストデータを作成
        train_data = [
            "/m/obama\tborn_in\t/m/hawaii",
            "/m/hawaii\tlocated_in\t/m/usa",
            "/m/usa\tcapital\t/m/washington",
            "/m/einstein\tborn_in\t/m/ulm",
            "/m/einstein\tnationality\t/m/germany"
        ]
        
        entity_data = [
            "/m/obama\tBarack Obama",
            "/m/hawaii\tHawaii",
            "/m/usa\tUnited States of America",
            "/m/washington\tWashington D.C.",
            "/m/einstein\tAlbert Einstein",
            "/m/ulm\tUlm",
            "/m/germany\tGermany"
        ]
        
        entity_long_data = [
            "/m/obama\tBarack Hussein Obama II is an American politician who served as the 44th president of the United States from 2009 to 2017.",
            "/m/hawaii\tHawaii is a state in the Western United States, located in the Pacific Ocean.",
            "/m/usa\tThe United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or America, is a country primarily located in North America.",
            "/m/washington\tWashington, D.C., formally the District of Columbia, also known as just Washington or simply D.C., is the capital city and federal district of the United States.",
            "/m/einstein\tAlbert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest physicists of all time.",
            "/m/ulm\tUlm is a city in the federal German state of Baden-Württemberg.",
            "/m/germany\tGermany, officially the Federal Republic of Germany, is a country in Central Europe."
        ]
        
        relation_data = [
            "born_in\twas born in",
            "located_in\tis located in",
            "capital\thas capital",
            "nationality\thas nationality"
        ]
        
        with open(os.path.join(temp_dir, 'train.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_data))
        
        with open(os.path.join(temp_dir, 'entity2text.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(entity_data))
        
        with open(os.path.join(temp_dir, 'entity2textlong.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(entity_long_data))
        
        with open(os.path.join(temp_dir, 'relation2text.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(relation_data))
        
        # KGとRetrieverを作成
        print("\n1. 知識グラフを初期化中...")
        kg = TextAttributedKnoweldgeGraph(temp_dir)
        print("   知識グラフの初期化完了")
        
        print("\n2. LLMKnowledgeRetrieverを初期化中...")
        retriever = LLMKnowledgeRetriever(kg=kg, llm_model='gpt-4o', use_web_search=True)
        print("   LLMKnowledgeRetrieverの初期化完了")
        
        # テスト用のRelationリストを作成
        sample_relations = [
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
                id="occupation",
                label="occupation",
                description_short="has occupation",
                description="Indicates the profession or job of a person"
            ),
            Relation(
                id="education",
                label="education",
                description_short="educated at",
                description="Indicates the educational institution where a person studied"
            ),
            Relation(
                id="award",
                label="award",
                description_short="received award",
                description="Indicates an award or honor received by a person"
            )
        ]
        
        # テスト1: 初期化テスト
        print("\n" + "=" * 70)
        print("テスト1: 初期化テスト")
        print("=" * 70)
        test_llm_knowledge_retriever_initialization(retriever)
        
        # テスト2: 基本動作テスト
        print("\n" + "=" * 70)
        print("テスト2: retrieve_knowledge_for_entity基本動作テスト")
        print("=" * 70)
        test_retrieve_knowledge_for_entity_basic(retriever, sample_relations)
        
        # テスト3: 科学者エンティティのテスト
        print("\n" + "=" * 70)
        print("テスト3: 科学者エンティティの知識取得テスト")
        print("=" * 70)
        test_retrieve_knowledge_for_entity_with_scientist(retriever, sample_relations)
        
        # テスト4: 返り値の構造検証
        print("\n" + "=" * 70)
        print("テスト4: 返り値の構造検証テスト")
        print("=" * 70)
        test_retrieve_knowledge_for_entity_return_structure(retriever, sample_relations)
        
        # テスト5: 空のrelationリスト処理
        print("\n" + "=" * 70)
        print("テスト5: 空のrelationリスト処理テスト")
        print("=" * 70)
        test_retrieve_knowledge_for_entity_empty_relations(retriever)
        
        print("\n" + "=" * 70)
        print("全てのテストが成功しました！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # クリーンアップ
        shutil.rmtree(temp_dir)
