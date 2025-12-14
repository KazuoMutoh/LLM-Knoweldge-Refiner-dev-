"""
ユニットテスト: triples_editor.py

トリプル追加ロジックの核心部分をテストします:
- TriplePattern のvariables()とinstantiate()
- _unify_constants と _extend_theta_with_triple
- _unify_head_with_triple
- _backtrack_patterns (conjunctive query処理)
- find_body_triples_for_head (ルールマッチング)
- add_triples_for_single_rule (統合処理)
"""

import pytest
from typing import Dict, List, Set
from simple_active_refine.triples_editor import (
    TriplePattern,
    Rule,
    _unify_constants,
    _extend_theta_with_triple,
    _unify_head_with_triple,
    _backtrack_patterns,
    find_body_triples_for_head,
    TripleIndex,
)

# ==========================
# TriplePattern のテスト
# ==========================

def test_triple_pattern_variables_no_variables():
    """変数を含まないパターン"""
    pat = TriplePattern("Obama", "nationality", "USA")
    assert pat.variables() == set()

def test_triple_pattern_variables_single():
    """単一変数を含むパターン"""
    pat = TriplePattern("?x", "nationality", "USA")
    assert pat.variables() == {"?x"}

def test_triple_pattern_variables_multiple():
    """複数変数を含むパターン"""
    pat = TriplePattern("?person", "?rel", "?country")
    assert pat.variables() == {"?person", "?rel", "?country"}

def test_triple_pattern_instantiate_no_variables():
    """変数なしパターンのインスタンス化"""
    pat = TriplePattern("Obama", "nationality", "USA")
    result = pat.instantiate({})
    assert result == ("Obama", "nationality", "USA")

def test_triple_pattern_instantiate_with_substitution():
    """変数をインスタンス化"""
    pat = TriplePattern("?person", "nationality", "?country")
    theta = {"?person": "Obama", "?country": "USA"}
    result = pat.instantiate(theta)
    assert result == ("Obama", "nationality", "USA")

def test_triple_pattern_instantiate_partial():
    """一部の変数のみインスタンス化（実装はすべての変数が必要）"""
    pat = TriplePattern("?person", "nationality", "?country")
    theta = {"?person": "Obama"}
    
    # instantiate() は全変数を要求するため、KeyError が発生することを確認
    with pytest.raises(KeyError):
        result = pat.instantiate(theta)

# ==========================
# _unify_constants のテスト
# ==========================

def test_unify_constants_both_constants_equal():
    """両方が定数で等しい"""
    assert _unify_constants("Obama", "Obama") is True

def test_unify_constants_both_constants_different():
    """両方が定数で異なる"""
    assert _unify_constants("Obama", "Trump") is False

def test_unify_constants_one_variable():
    """片方が変数"""
    assert _unify_constants("?x", "Obama") is True
    assert _unify_constants("Obama", "?x") is True

def test_unify_constants_both_variables():
    """両方が変数"""
    assert _unify_constants("?x", "?y") is True

# ==========================
# _extend_theta_with_triple のテスト
# ==========================

def test_extend_theta_empty():
    """空の代入でパターンマッチ"""
    pat = TriplePattern("?person", "nationality", "USA")
    triple = ("Obama", "nationality", "USA")
    result = _extend_theta_with_triple({}, pat, triple)
    assert result == {"?person": "Obama"}

def test_extend_theta_consistent():
    """既存の代入と一貫性がある"""
    pat = TriplePattern("?person", "born_in", "?place")
    triple = ("Obama", "born_in", "Hawaii")
    theta = {"?person": "Obama"}
    result = _extend_theta_with_triple(theta, pat, triple)
    assert result == {"?person": "Obama", "?place": "Hawaii"}

def test_extend_theta_inconsistent():
    """既存の代入と矛盾"""
    pat = TriplePattern("?person", "nationality", "USA")
    triple = ("Trump", "nationality", "USA")
    theta = {"?person": "Obama"}
    result = _extend_theta_with_triple(theta, pat, triple)
    assert result is None

def test_extend_theta_constant_mismatch():
    """定数が一致しない"""
    pat = TriplePattern("Obama", "nationality", "USA")
    triple = ("Trump", "nationality", "USA")
    result = _extend_theta_with_triple({}, pat, triple)
    assert result is None

# ==========================
# _unify_head_with_triple のテスト
# ==========================

def test_unify_head_simple():
    """ヘッドパターンとトリプルの単純な統一"""
    head_pat = TriplePattern("?person", "nationality", "?country")
    head_triple = ("Obama", "nationality", "USA")
    theta = _unify_head_with_triple(head_pat, head_triple)
    assert theta == {"?person": "Obama", "?country": "USA"}

def test_unify_head_constant_match():
    """定数を含むヘッドパターン"""
    head_pat = TriplePattern("?person", "nationality", "USA")
    head_triple = ("Obama", "nationality", "USA")
    theta = _unify_head_with_triple(head_pat, head_triple)
    assert theta == {"?person": "Obama"}

def test_unify_head_constant_mismatch():
    """定数が一致しない"""
    head_pat = TriplePattern("?person", "nationality", "USA")
    head_triple = ("Obama", "nationality", "UK")
    theta = _unify_head_with_triple(head_pat, head_triple)
    assert theta is None

def test_unify_head_all_constants():
    """すべて定数で完全一致"""
    head_pat = TriplePattern("Obama", "nationality", "USA")
    head_triple = ("Obama", "nationality", "USA")
    theta = _unify_head_with_triple(head_pat, head_triple)
    assert theta == {}

def test_unify_head_all_constants_mismatch():
    """すべて定数だが不一致"""
    head_pat = TriplePattern("Obama", "nationality", "USA")
    head_triple = ("Trump", "nationality", "USA")
    theta = _unify_head_with_triple(head_pat, head_triple)
    assert theta is None

# ==========================
# TripleIndex のテスト
# ==========================

def test_triple_index_match_all_bound():
    """すべてバインド済みのパターンマッチ"""
    triples = [
        ("Obama", "nationality", "USA"),
        ("Trump", "nationality", "USA"),
        ("Obama", "born_in", "Hawaii"),
    ]
    idx = TripleIndex(triples)
    
    pat = TriplePattern("Obama", "nationality", "USA")
    matches = list(idx.match_pattern(pat, {}))
    assert matches == [("Obama", "nationality", "USA")]

def test_triple_index_match_by_sp():
    """s, p バインド, o は変数"""
    triples = [
        ("Obama", "nationality", "USA"),
        ("Obama", "nationality", "Kenya"),  # hypothetical
    ]
    idx = TripleIndex(triples)
    
    pat = TriplePattern("Obama", "nationality", "?country")
    matches = list(idx.match_pattern(pat, {}))
    assert set(matches) == {("Obama", "nationality", "USA"), ("Obama", "nationality", "Kenya")}

def test_triple_index_match_by_p():
    """p のみバインド"""
    triples = [
        ("Obama", "nationality", "USA"),
        ("Trump", "nationality", "USA"),
        ("Biden", "born_in", "Pennsylvania"),
    ]
    idx = TripleIndex(triples)
    
    pat = TriplePattern("?person", "nationality", "?country")
    matches = list(idx.match_pattern(pat, {}))
    assert set(matches) == {("Obama", "nationality", "USA"), ("Trump", "nationality", "USA")}

def test_triple_index_match_with_theta():
    """既存代入を使ったマッチング"""
    triples = [
        ("Obama", "nationality", "USA"),
        ("Trump", "nationality", "USA"),
    ]
    idx = TripleIndex(triples)
    
    pat = TriplePattern("?person", "nationality", "USA")
    theta = {"?person": "Obama"}
    matches = list(idx.match_pattern(pat, theta))
    assert matches == [("Obama", "nationality", "USA")]

# ==========================
# _backtrack_patterns のテスト
# ==========================

def test_backtrack_single_pattern():
    """単一パターンのバックトラック"""
    patterns = [TriplePattern("?person", "nationality", "USA")]
    triples = [
        ("Obama", "nationality", "USA"),
        ("Trump", "nationality", "USA"),
    ]
    idx = TripleIndex(triples)
    
    results = list(_backtrack_patterns(patterns, idx, {}))
    assert len(results) == 2
    
    thetas = [theta for theta, _ in results]
    assert {"?person": "Obama"} in thetas
    assert {"?person": "Trump"} in thetas

def test_backtrack_two_patterns_join():
    """2つのパターンの結合"""
    # ?person が nationality=USA かつ born_in=Hawaii
    patterns = [
        TriplePattern("?person", "nationality", "USA"),
        TriplePattern("?person", "born_in", "Hawaii"),
    ]
    triples = [
        ("Obama", "nationality", "USA"),
        ("Trump", "nationality", "USA"),
        ("Obama", "born_in", "Hawaii"),
    ]
    idx = TripleIndex(triples)
    
    results = list(_backtrack_patterns(patterns, idx, {}))
    assert len(results) == 1
    
    theta, used = results[0]
    assert theta == {"?person": "Obama"}
    assert set(used) == {("Obama", "nationality", "USA"), ("Obama", "born_in", "Hawaii")}

def test_backtrack_no_match():
    """マッチするものがない"""
    patterns = [
        TriplePattern("?person", "nationality", "USA"),
        TriplePattern("?person", "born_in", "UK"),  # USA国籍だがUK生まれ（矛盾）
    ]
    triples = [
        ("Obama", "nationality", "USA"),
        ("Obama", "born_in", "Hawaii"),
    ]
    idx = TripleIndex(triples)
    
    results = list(_backtrack_patterns(patterns, idx, {}))
    assert len(results) == 0

def test_backtrack_three_patterns():
    """3つのパターンの結合"""
    patterns = [
        TriplePattern("?person", "nationality", "?country"),
        TriplePattern("?person", "born_in", "?place"),
        TriplePattern("?place", "located_in", "?country"),
    ]
    triples = [
        ("Obama", "nationality", "USA"),
        ("Obama", "born_in", "Hawaii"),
        ("Hawaii", "located_in", "USA"),
        ("Trump", "nationality", "USA"),
        ("Trump", "born_in", "NYC"),
        ("NYC", "located_in", "USA"),
    ]
    idx = TripleIndex(triples)
    
    results = list(_backtrack_patterns(patterns, idx, {}))
    assert len(results) == 2
    
    # Obama と Trump の両方が一致すべき
    thetas = [theta for theta, _ in results]
    assert {"?person": "Obama", "?country": "USA", "?place": "Hawaii"} in thetas
    assert {"?person": "Trump", "?country": "USA", "?place": "NYC"} in thetas

# ==========================
# find_body_triples_for_head のテスト
# ==========================

def test_find_body_triples_simple_rule():
    """単純なルールでボディトリプルを発見"""
    # ルール: nationality(?x, ?y) :- born_in(?x, ?z), located_in(?z, ?y)
    rule = Rule(
        head=TriplePattern("?x", "nationality", "?y"),
        body=[
            TriplePattern("?x", "born_in", "?z"),
            TriplePattern("?z", "located_in", "?y"),
        ],
        support=10,
        std_conf=0.8,
        pca_conf=0.9,
        head_coverage=0.7,
        body_size=2,
        pca_body_size=2,
    )
    
    head_triple = ("Obama", "nationality", "USA")
    
    candidates = [
        ("Obama", "born_in", "Hawaii"),
        ("Hawaii", "located_in", "USA"),
        ("Trump", "born_in", "NYC"),
        ("NYC", "located_in", "USA"),
    ]
    
    body_triples = find_body_triples_for_head(head_triple, [rule], candidates)
    
    assert set(body_triples) == {
        ("Obama", "born_in", "Hawaii"),
        ("Hawaii", "located_in", "USA"),
    }

def test_find_body_triples_multiple_rules():
    """複数のルールを試行"""
    rule1 = Rule(
        head=TriplePattern("?x", "nationality", "?y"),
        body=[TriplePattern("?x", "born_in", "?y")],
        support=5,
        std_conf=0.7,
        pca_conf=0.8,
        head_coverage=0.6,
        body_size=1,
        pca_body_size=1,
    )
    
    rule2 = Rule(
        head=TriplePattern("?x", "nationality", "?y"),
        body=[TriplePattern("?x", "citizen_of", "?y")],
        support=8,
        std_conf=0.9,
        pca_conf=0.95,
        head_coverage=0.8,
        body_size=1,
        pca_body_size=1,
    )
    
    head_triple = ("Obama", "nationality", "USA")
    
    candidates = [
        ("Obama", "born_in", "USA"),
        ("Obama", "citizen_of", "USA"),
        ("Trump", "born_in", "USA"),
    ]
    
    body_triples = find_body_triples_for_head(head_triple, [rule1, rule2], candidates)
    
    # 両方のルールにマッチ
    assert set(body_triples) == {
        ("Obama", "born_in", "USA"),
        ("Obama", "citizen_of", "USA"),
    }

def test_find_body_triples_no_match():
    """マッチするボディトリプルがない"""
    rule = Rule(
        head=TriplePattern("?x", "nationality", "?y"),
        body=[TriplePattern("?x", "born_in", "?z"), TriplePattern("?z", "located_in", "?y")],
        support=10,
        std_conf=0.8,
        pca_conf=0.9,
        head_coverage=0.7,
        body_size=2,
        pca_body_size=2,
    )
    
    head_triple = ("Obama", "nationality", "USA")
    
    candidates = [
        ("Trump", "born_in", "NYC"),  # 異なる人物
        ("NYC", "located_in", "USA"),
    ]
    
    body_triples = find_body_triples_for_head(head_triple, [rule], candidates)
    assert body_triples == []

def test_find_body_triples_relation_mismatch():
    """ヘッドのリレーションが一致しない"""
    rule = Rule(
        head=TriplePattern("?x", "nationality", "?y"),
        body=[TriplePattern("?x", "born_in", "?y")],
        support=5,
        std_conf=0.7,
        pca_conf=0.8,
        head_coverage=0.6,
        body_size=1,
        pca_body_size=1,
    )
    
    head_triple = ("Obama", "occupation", "President")  # 異なるリレーション
    
    candidates = [
        ("Obama", "born_in", "Hawaii"),
    ]
    
    body_triples = find_body_triples_for_head(head_triple, [rule], candidates)
    assert body_triples == []

def test_find_body_triples_variable_relation_in_head():
    """ヘッドのリレーションが変数の場合"""
    rule = Rule(
        head=TriplePattern("?x", "?rel", "?y"),
        body=[TriplePattern("?x", "born_in", "?y")],
        support=5,
        std_conf=0.7,
        pca_conf=0.8,
        head_coverage=0.6,
        body_size=1,
        pca_body_size=1,
    )
    
    head_triple = ("Obama", "nationality", "USA")
    
    candidates = [
        ("Obama", "born_in", "USA"),
    ]
    
    body_triples = find_body_triples_for_head(head_triple, [rule], candidates)
    assert body_triples == [("Obama", "born_in", "USA")]

# ==========================
# 統合テスト（現実的なシナリオ）
# ==========================

def test_realistic_nationality_rule():
    """現実的な国籍ルールのテスト"""
    # FB15k-237 の /people/person/nationality に似たルール
    # nationality(?person, ?country) :- 
    #   place_of_birth(?person, ?city), 
    #   location/location/containedby(?city, ?country)
    
    rule = Rule(
        head=TriplePattern("?a", "/people/person/nationality", "?b"),
        body=[
            TriplePattern("?a", "/people/person/place_of_birth", "?c"),
            TriplePattern("?c", "/location/location/containedby", "?b"),
        ],
        support=150,
        std_conf=0.75,
        pca_conf=0.85,
        head_coverage=0.7,
        body_size=2,
        pca_body_size=2,
    )
    
    head_triple = ("/m/obama", "/people/person/nationality", "/m/usa")
    
    # ルールは2つのパターンなので、直接的な containedby のみがマッチ
    candidates = [
        ("/m/obama", "/people/person/place_of_birth", "/m/honolulu"),
        ("/m/honolulu", "/location/location/containedby", "/m/usa"),  # 直接USA
        ("/m/trump", "/people/person/place_of_birth", "/m/nyc"),
        ("/m/nyc", "/location/location/containedby", "/m/usa"),
    ]
    
    body_triples = find_body_triples_for_head(head_triple, [rule], candidates)
    
    # Obamaの出生地とその直接の上位地域を発見
    assert set(body_triples) == {
        ("/m/obama", "/people/person/place_of_birth", "/m/honolulu"),
        ("/m/honolulu", "/location/location/containedby", "/m/usa"),
    }

def test_complex_multi_hop_rule():
    """複雑な多段階ルール"""
    # nationality(?person, ?country) :- 
    #   education(?person, ?school),
    #   location(?school, ?city),
    #   containedby(?city, ?country)
    
    rule = Rule(
        head=TriplePattern("?person", "nationality", "?country"),
        body=[
            TriplePattern("?person", "education", "?school"),
            TriplePattern("?school", "location", "?city"),
            TriplePattern("?city", "containedby", "?country"),
        ],
        support=50,
        std_conf=0.6,
        pca_conf=0.7,
        head_coverage=0.5,
        body_size=3,
        pca_body_size=3,
    )
    
    head_triple = ("Obama", "nationality", "USA")
    
    # 3ホップのチェーンを直接含む候補
    candidates = [
        ("Obama", "education", "Harvard"),
        ("Harvard", "location", "Cambridge"),
        ("Cambridge", "containedby", "USA"),  # 直接USA
        ("Trump", "education", "Wharton"),
        ("Wharton", "location", "Philadelphia"),
        ("Philadelphia", "containedby", "USA"),
    ]
    
    body_triples = find_body_triples_for_head(head_triple, [rule], candidates)
    
    assert set(body_triples) == {
        ("Obama", "education", "Harvard"),
        ("Harvard", "location", "Cambridge"),
        ("Cambridge", "containedby", "USA"),
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
