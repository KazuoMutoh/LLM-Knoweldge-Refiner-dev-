from simple_active_refine.triples_editor import TriplePattern, TripleIndex, Rule, count_witnesses_for_head, supports_head


def test_count_witnesses_for_head_basic():
    rule = Rule(
        head=TriplePattern("?x", "r*", "?y"),
        body=[TriplePattern("?x", "p1", "?z"), TriplePattern("?z", "p2", "?y")],
        support=None,
        std_conf=None,
        pca_conf=None,
        head_coverage=None,
        body_size=None,
        pca_body_size=None,
    )
    candidates = [
        ("a", "p1", "b"),
        ("b", "p2", "c"),
        ("a", "p1", "d"),
        ("d", "p2", "c"),
    ]
    head = ("a", "r*", "c")
    idx = TripleIndex(candidates)
    count = count_witnesses_for_head(head, rule, idx)
    assert count == 2  # (a,p1,b)-(b,p2,c) and (a,p1,d)-(d,p2,c)
    assert supports_head(head, rule, idx)


def test_count_witnesses_respects_relation():
    rule = Rule(
        head=TriplePattern("?x", "r*", "?y"),
        body=[],
        support=None,
        std_conf=None,
        pca_conf=None,
        head_coverage=None,
        body_size=None,
        pca_body_size=None,
    )
    head = ("a", "other", "b")
    assert count_witnesses_for_head(head, rule, [], max_witness=1) == 0
    assert not supports_head(head, rule, [])
