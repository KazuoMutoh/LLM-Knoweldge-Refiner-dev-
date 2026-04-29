from simple_active_refine.amie import AmieRule, TriplePattern
from simple_active_refine.arm_builder import ArmBuilderConfig, build_initial_arms


def _rule(body):
    return AmieRule(
        head=TriplePattern("?x", "r*", "?y"),
        body=body,
        support=None,
        std_conf=None,
        pca_conf=None,
        head_coverage=None,
        body_size=None,
        pca_body_size=None,
        raw="test",
    )


def test_build_initial_arms_pair_selection():
    # two rules sharing one supported target, one rule isolated
    r1 = _rule([TriplePattern("?x", "p1", "?z"), TriplePattern("?z", "p2", "?y")])
    r2 = _rule([TriplePattern("?x", "p1", "?z"), TriplePattern("?z", "p3", "?y")])
    r3 = _rule([TriplePattern("?x", "p4", "?y")])

    target_triples = [("a", "r*", "c"), ("d", "r*", "e")]
    candidates = [
        ("a", "p1", "b"), ("b", "p2", "c"),  # supports r1
        ("a", "p1", "b"), ("b", "p3", "c"),  # supports r2
        ("d", "p4", "e"),  # supports r3
    ]

    cfg = ArmBuilderConfig(k_pairs=1)
    arms = build_initial_arms([r1, r2, r3], target_triples, candidates, cfg)

    singleton_keys = [a.rule_keys for a in arms if a.metadata.get("kind") == "singleton"]
    assert len(singleton_keys) == 3

    pair_arms = [a for a in arms if a.metadata.get("kind") == "pair"]
    # only one pair kept (k_pairs=1), should be r1-r2 because they share a supported target
    assert len(pair_arms) == 1
    keys = set(pair_arms[0].rule_keys)
    assert len(keys) == 2


def test_build_initial_arms_pair_selection_can_use_train_support_triples():
    # Candidate triples support rules on disjoint targets -> no pair.
    # Train-support triples support two rules on the same target -> pair should appear.
    r1 = _rule([TriplePattern("?x", "p1", "?z"), TriplePattern("?z", "p2", "?y")])
    r2 = _rule([TriplePattern("?x", "p1", "?z"), TriplePattern("?z", "p3", "?y")])

    target_triples = [("a", "r*", "c"), ("d", "r*", "e")]

    candidates = [
        ("a", "p1", "b"), ("b", "p2", "c"),  # supports r1 on target a
        ("d", "p1", "b"), ("b", "p3", "e"),  # supports r2 on target d (disjoint)
    ]
    train_support = [
        ("a", "p1", "b"), ("b", "p2", "c"),  # supports r1 on target a
        ("a", "p1", "b"), ("b", "p3", "c"),  # supports r2 on target a (shared)
    ]

    cfg = ArmBuilderConfig(k_pairs=1, pair_support_source="train")
    arms = build_initial_arms([r1, r2], target_triples, candidates, cfg, pair_support_triples=train_support)

    pair_arms = [a for a in arms if a.metadata.get("kind") == "pair"]
    assert len(pair_arms) == 1

