"""
統合テスト: add_triples_for_single_rule

実際のファイルを使って、単一ルールに基づくトリプル追加処理をテストします。
"""

import pytest
import os
import tempfile
import shutil
from typing import List, Tuple
from simple_active_refine.triples_editor import (
    TriplePattern,
    Rule,
    add_triples_for_single_rule,
    Triple,
)

@pytest.fixture
def temp_data_dir():
    """一時的なテストデータディレクトリを作成"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def create_test_data(temp_dir: str, 
                     train_triples: List[Tuple[str, str, str]],
                     removed_triples: List[Tuple[str, str, str]]):
    """テスト用のトリプルファイルを作成"""
    # train.txt
    train_path = os.path.join(temp_dir, 'train.txt')
    with open(train_path, 'w') as f:
        for h, r, t in train_triples:
            f.write(f'{h}\t{r}\t{t}\n')
    
    # train_removed.txt
    removed_path = os.path.join(temp_dir, 'train_removed.txt')
    with open(removed_path, 'w') as f:
        for h, r, t in removed_triples:
            f.write(f'{h}\t{r}\t{t}\n')
    
    return train_path, removed_path


def test_add_triples_simple_rule(temp_data_dir):
    """単純なルールでトリプルが正しく追加される"""
    # テストデータ作成
    train_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
        ("/m/trump", "/people/person/nationality", "/m/usa"),
    ]
    
    removed_triples = [
        # Obamaの出生地情報（これが追加されるべき）
        ("/m/obama", "/people/person/place_of_birth", "/m/honolulu"),
        ("/m/honolulu", "/location/location/containedby", "/m/usa"),
        # Trumpの出生地情報
        ("/m/trump", "/people/person/place_of_birth", "/m/nyc"),
        ("/m/nyc", "/location/location/containedby", "/m/usa"),
        # ノイズデータ
        ("/m/biden", "/people/person/education", "/m/delaware_univ"),
    ]
    
    create_test_data(temp_data_dir, train_triples, removed_triples)
    
    # ルール: nationality(?x, ?y) :- place_of_birth(?x, ?z), containedby(?z, ?y)
    rule = Rule(
        head=TriplePattern("?x", "/people/person/nationality", "?y"),
        body=[
            TriplePattern("?x", "/people/person/place_of_birth", "?z"),
            TriplePattern("?z", "/location/location/containedby", "?y"),
        ],
        support=100,
        std_conf=0.8,
        pca_conf=0.85,
        head_coverage=0.7,
        body_size=2,
        pca_body_size=2,
    )
    
    # 対象トリプル
    target_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
    ]
    
    # 実行
    added_triples, details = add_triples_for_single_rule(
        dir_triples=temp_data_dir,
        rule=[rule],
        target_triples=target_triples,
    )
    
    # 検証
    assert details['num_targets'] == 1
    assert details['total_added'] > 0
    
    # Obamaの出生地関連トリプルが含まれているか
    added_set = set(added_triples)
    assert ("/m/obama", "/people/person/place_of_birth", "/m/honolulu") in added_set
    assert ("/m/honolulu", "/location/location/containedby", "/m/usa") in added_set
    
    # 関連トリプルも追加されるので、総数は2以上になる
    assert len(added_triples) >= 2


def test_add_triples_multiple_targets(temp_data_dir):
    """複数のターゲットトリプルに対して正しく追加される"""
    train_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
        ("/m/trump", "/people/person/nationality", "/m/usa"),
    ]
    
    removed_triples = [
        ("/m/obama", "/people/person/place_of_birth", "/m/honolulu"),
        ("/m/honolulu", "/location/location/containedby", "/m/usa"),
        ("/m/trump", "/people/person/place_of_birth", "/m/nyc"),
        ("/m/nyc", "/location/location/containedby", "/m/usa"),
    ]
    
    create_test_data(temp_data_dir, train_triples, removed_triples)
    
    rule = Rule(
        head=TriplePattern("?x", "/people/person/nationality", "?y"),
        body=[
            TriplePattern("?x", "/people/person/place_of_birth", "?z"),
            TriplePattern("?z", "/location/location/containedby", "?y"),
        ],
        support=100,
        std_conf=0.8,
        pca_conf=0.85,
        head_coverage=0.7,
        body_size=2,
        pca_body_size=2,
    )
    
    target_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
        ("/m/trump", "/people/person/nationality", "/m/usa"),
    ]
    
    added_triples, details = add_triples_for_single_rule(
        dir_triples=temp_data_dir,
        rule=[rule],
        target_triples=target_triples,
    )
    
    # 検証
    assert details['num_targets'] == 2
    assert len(details['added_triples_by_target']) == 2
    
    # 両方のターゲットに対してトリプルが見つかる
    assert len(details['added_triples_by_target'][0]['triples_to_be_added']) > 0
    assert len(details['added_triples_by_target'][1]['triples_to_be_added']) > 0
    
    added_set = set(added_triples)
    # Obama関連
    assert ("/m/obama", "/people/person/place_of_birth", "/m/honolulu") in added_set
    # Trump関連
    assert ("/m/trump", "/people/person/place_of_birth", "/m/nyc") in added_set


def test_add_triples_no_match(temp_data_dir):
    """マッチするトリプルがない場合は空リストを返す"""
    train_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
    ]
    
    removed_triples = [
        # Obamaとは関係ないトリプル
        ("/m/trump", "/people/person/place_of_birth", "/m/nyc"),
        ("/m/nyc", "/location/location/containedby", "/m/usa"),
    ]
    
    create_test_data(temp_data_dir, train_triples, removed_triples)
    
    rule = Rule(
        head=TriplePattern("?x", "/people/person/nationality", "?y"),
        body=[
            TriplePattern("?x", "/people/person/place_of_birth", "?z"),
            TriplePattern("?z", "/location/location/containedby", "?y"),
        ],
        support=100,
        std_conf=0.8,
        pca_conf=0.85,
        head_coverage=0.7,
        body_size=2,
        pca_body_size=2,
    )
    
    target_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
    ]
    
    added_triples, details = add_triples_for_single_rule(
        dir_triples=temp_data_dir,
        rule=[rule],
        target_triples=target_triples,
    )
    
    # 検証
    assert details['num_targets'] == 1
    assert details['total_added'] == 0
    assert len(added_triples) == 0
    assert len(details['added_triples_by_target'][0]['triples_to_be_added']) == 0


def test_add_triples_partial_match(temp_data_dir):
    """一部のターゲットのみマッチする場合"""
    train_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
        ("/m/biden", "/people/person/nationality", "/m/usa"),
    ]
    
    removed_triples = [
        # Obamaのみ完全なパターンを持つ
        ("/m/obama", "/people/person/place_of_birth", "/m/honolulu"),
        ("/m/honolulu", "/location/location/containedby", "/m/usa"),
        # Bidenは place_of_birth のみ（containedby が欠落）
        ("/m/biden", "/people/person/place_of_birth", "/m/scranton"),
    ]
    
    create_test_data(temp_data_dir, train_triples, removed_triples)
    
    rule = Rule(
        head=TriplePattern("?x", "/people/person/nationality", "?y"),
        body=[
            TriplePattern("?x", "/people/person/place_of_birth", "?z"),
            TriplePattern("?z", "/location/location/containedby", "?y"),
        ],
        support=100,
        std_conf=0.8,
        pca_conf=0.85,
        head_coverage=0.7,
        body_size=2,
        pca_body_size=2,
    )
    
    target_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
        ("/m/biden", "/people/person/nationality", "/m/usa"),
    ]
    
    added_triples, details = add_triples_for_single_rule(
        dir_triples=temp_data_dir,
        rule=[rule],
        target_triples=target_triples,
    )
    
    # 検証
    assert details['num_targets'] == 2
    
    # Obamaのみマッチ
    obama_result = next(d for d in details['added_triples_by_target'] 
                       if d['target_triple'][0] == "/m/obama")
    biden_result = next(d for d in details['added_triples_by_target'] 
                       if d['target_triple'][0] == "/m/biden")
    
    assert len(obama_result['triples_to_be_added']) > 0
    assert len(biden_result['triples_to_be_added']) == 0
    
    # Obamaのトリプルのみ追加される
    added_set = set(added_triples)
    assert ("/m/obama", "/people/person/place_of_birth", "/m/honolulu") in added_set


def test_add_triples_complex_rule(temp_data_dir):
    """3つのbodyパターンを持つ複雑なルール"""
    train_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
    ]
    
    removed_triples = [
        # 3ホップチェーン
        ("/m/obama", "/people/person/education", "/m/harvard"),
        ("/m/harvard", "/education/institution/location", "/m/cambridge"),
        ("/m/cambridge", "/location/location/containedby", "/m/usa"),
    ]
    
    create_test_data(temp_data_dir, train_triples, removed_triples)
    
    # 3パターンルール
    rule = Rule(
        head=TriplePattern("?x", "/people/person/nationality", "?y"),
        body=[
            TriplePattern("?x", "/people/person/education", "?school"),
            TriplePattern("?school", "/education/institution/location", "?city"),
            TriplePattern("?city", "/location/location/containedby", "?y"),
        ],
        support=50,
        std_conf=0.6,
        pca_conf=0.7,
        head_coverage=0.5,
        body_size=3,
        pca_body_size=3,
    )
    
    target_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
    ]
    
    added_triples, details = add_triples_for_single_rule(
        dir_triples=temp_data_dir,
        rule=[rule],
        target_triples=target_triples,
    )
    
    # 検証
    assert details['total_added'] > 0
    
    # 3つすべてのパターントリプルが含まれる
    added_set = set(added_triples)
    assert ("/m/obama", "/people/person/education", "/m/harvard") in added_set
    assert ("/m/harvard", "/education/institution/location", "/m/cambridge") in added_set
    assert ("/m/cambridge", "/location/location/containedby", "/m/usa") in added_set


def test_add_triples_with_related_triples(temp_data_dir):
    """関連トリプル（接続されたトリプル）も追加される"""
    train_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
    ]
    
    removed_triples = [
        # ルールにマッチするトリプル
        ("/m/obama", "/people/person/place_of_birth", "/m/honolulu"),
        ("/m/honolulu", "/location/location/containedby", "/m/usa"),
        # Honoluluに接続する追加情報（関連トリプル）
        ("/m/honolulu", "/location/location/population", "350000"),
        ("/m/honolulu", "/location/location/timezone", "/m/hst"),
    ]
    
    create_test_data(temp_data_dir, train_triples, removed_triples)
    
    rule = Rule(
        head=TriplePattern("?x", "/people/person/nationality", "?y"),
        body=[
            TriplePattern("?x", "/people/person/place_of_birth", "?z"),
            TriplePattern("?z", "/location/location/containedby", "?y"),
        ],
        support=100,
        std_conf=0.8,
        pca_conf=0.85,
        head_coverage=0.7,
        body_size=2,
        pca_body_size=2,
    )
    
    target_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
    ]
    
    added_triples, details = add_triples_for_single_rule(
        dir_triples=temp_data_dir,
        rule=[rule],
        target_triples=target_triples,
    )
    
    # 検証：関連トリプルも追加される
    added_set = set(added_triples)
    
    # ルールマッチしたトリプル
    assert ("/m/obama", "/people/person/place_of_birth", "/m/honolulu") in added_set
    assert ("/m/honolulu", "/location/location/containedby", "/m/usa") in added_set
    
    # Honoluluに接続する関連トリプルも追加される
    assert ("/m/honolulu", "/location/location/population", "350000") in added_set
    assert ("/m/honolulu", "/location/location/timezone", "/m/hst") in added_set
    
    # ルールマッチ(2) + 関連(2) = 最低4個
    assert len(added_triples) >= 4


def test_add_triples_details_structure(temp_data_dir):
    """詳細情報の構造が正しい"""
    train_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
    ]
    
    removed_triples = [
        ("/m/obama", "/people/person/place_of_birth", "/m/honolulu"),
        ("/m/honolulu", "/location/location/containedby", "/m/usa"),
    ]
    
    create_test_data(temp_data_dir, train_triples, removed_triples)
    
    rule = Rule(
        head=TriplePattern("?x", "/people/person/nationality", "?y"),
        body=[
            TriplePattern("?x", "/people/person/place_of_birth", "?z"),
            TriplePattern("?z", "/location/location/containedby", "?y"),
        ],
        support=100,
        std_conf=0.8,
        pca_conf=0.85,
        head_coverage=0.7,
        body_size=2,
        pca_body_size=2,
    )
    
    target_triples = [
        ("/m/obama", "/people/person/nationality", "/m/usa"),
    ]
    
    added_triples, details = add_triples_for_single_rule(
        dir_triples=temp_data_dir,
        rule=[rule],
        target_triples=target_triples,
    )
    
    # 詳細情報の構造検証
    assert 'target_triples' in details
    assert 'added_triples_by_target' in details
    assert 'total_added' in details
    assert 'num_targets' in details
    
    assert details['target_triples'] == target_triples
    assert details['num_targets'] == len(target_triples)
    assert details['total_added'] == len(added_triples)
    
    # added_triples_by_targetの構造
    assert len(details['added_triples_by_target']) == 1
    first_entry = details['added_triples_by_target'][0]
    assert 'target_triple' in first_entry
    assert 'triples_to_be_added' in first_entry
    assert first_entry['target_triple'] == target_triples[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
