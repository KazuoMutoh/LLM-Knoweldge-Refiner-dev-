import pytest

from simple_active_refine.amie import AmieRule, TriplePattern
from simple_active_refine.rule_history import RuleEvaluationRecord, RuleHistory
from simple_active_refine.rule_selector import (
    LLMPolicyRuleSelector,
    UCBRuleSelector,
    EpsilonGreedyRuleSelector,
    RandomRuleSelector,
    RuleWithId,
    SelectionPolicy,
)


def _make_rule(body_rel: str, head_rel: str, pca_conf: float | None = None) -> AmieRule:
    head = TriplePattern("?x", head_rel, "?y")
    body = [TriplePattern("?x", body_rel, "?y")]
    return AmieRule(
        head=head,
        body=body,
        support=None,
        std_conf=None,
        pca_conf=pca_conf,
        head_coverage=None,
        body_size=None,
        pca_body_size=None,
        raw=f"{body_rel}->{head_rel}",
    )


def _add_history(rule_history: RuleHistory, rule_id: str, rule: AmieRule, score_changes: list[float]):
    record = RuleEvaluationRecord(
        iteration=1,
        rule_id=rule_id,
        rule=rule,
        target_triples=[("h", "r", "t")],
        added_triples=[("a", "b", "c")],
        score_changes=score_changes,
        mean_score_change=sum(score_changes) / len(score_changes),
        std_score_change=0.0,
        positive_changes=len([s for s in score_changes if s > 0]),
        negative_changes=len([s for s in score_changes if s < 0]),
    )
    rule_history.add_record(record)


def test_llm_policy_uses_history_and_mocked_llm(monkeypatch):
    # Prepare rules and history
    r1 = RuleWithId.create(_make_rule("/b1", "/h", pca_conf=0.1), rule_id="r1")
    r2 = RuleWithId.create(_make_rule("/b2", "/h", pca_conf=0.2), rule_id="r2")
    r3 = RuleWithId.create(_make_rule("/b3", "/h", pca_conf=0.3), rule_id="r3")
    history = RuleHistory()
    _add_history(history, "r1", r1.rule, [0.1, 0.2])  # r1 tried and positive
    # r2 untried

    selector = LLMPolicyRuleSelector(history=history, chat_model="gpt-4o", temperature=0.0)

    # Mock LLM output to select r2 (untried) then r1
    class _DummyLLM:
        def invoke(self, prompt):
            return SelectionPolicy(
                reasoning="mock",
                policy_text="mock policy",
                selected_rule_ids=["r2", "r1"],
                rationale_per_rule={"r2": "explore", "r1": "exploit"},
            )

    monkeypatch.setattr(selector, "structured_llm", _DummyLLM())

    selected, policy = selector.select_rules([r1, r2, r3], k=2, iteration=2)

    assert [r.rule_id for r in selected] == ["r2", "r1"]
    assert policy == "mock policy"


def test_llm_policy_iteration0_respects_pool_order():
    r1 = RuleWithId.create(_make_rule("/b1", "/h", pca_conf=0.1), rule_id="r1")
    r2 = RuleWithId.create(_make_rule("/b2", "/h", pca_conf=0.9), rule_id="r2")
    selector = LLMPolicyRuleSelector(history=RuleHistory(), chat_model="gpt-4o", temperature=0.0)

    selected, _ = selector.select_rules([r1, r2], k=1, iteration=0)
    # Selection should follow the pre-ranked pool (no re-sorting by pca_conf)
    assert [r.rule_id for r in selected] == ["r1"]


def test_ucb_prefers_untried_rules(monkeypatch):
    r_tried = RuleWithId.create(_make_rule("/b1", "/h"), rule_id="r_tried")
    r_new = RuleWithId.create(_make_rule("/b2", "/h"), rule_id="r_new")

    history = RuleHistory()
    _add_history(history, "r_tried", r_tried.rule, [0.05, 0.1])

    selector = UCBRuleSelector(history=history, exploration_param=1.0)

    selected, _ = selector.select_rules([r_tried, r_new], k=1, iteration=3)

    # Untried rule gets infinite UCB bonus and should be picked first
    assert [r.rule_id for r in selected] == ["r_new"]


def test_epsilon_greedy_exploit_chooses_best_mean(monkeypatch):
    r_low = RuleWithId.create(_make_rule("/b1", "/h"), rule_id="r_low")
    r_high = RuleWithId.create(_make_rule("/b2", "/h"), rule_id="r_high")

    history = RuleHistory()
    _add_history(history, "r_low", r_low.rule, [0.05])
    _add_history(history, "r_high", r_high.rule, [0.5])

    selector = EpsilonGreedyRuleSelector(history=history, epsilon=0.0)

    selected, _ = selector.select_rules([r_low, r_high], k=1, iteration=2)

    # epsilon=0 forces exploitation; highest mean score wins
    assert [r.rule_id for r in selected] == ["r_high"]


def test_epsilon_greedy_explore_uses_random(monkeypatch):
    r1 = RuleWithId.create(_make_rule("/b1", "/h"), rule_id="r1")
    r2 = RuleWithId.create(_make_rule("/b2", "/h"), rule_id="r2")

    selector = EpsilonGreedyRuleSelector(history=RuleHistory(), epsilon=1.0)

    # Force exploration and deterministic choice
    monkeypatch.setattr("simple_active_refine.rule_selector.random.random", lambda: 0.0)
    monkeypatch.setattr("simple_active_refine.rule_selector.random.choice", lambda seq: seq[0])

    selected, _ = selector.select_rules([r1, r2], k=1, iteration=1)

    assert [r.rule_id for r in selected] == ["r1"]


def test_random_selector_uses_random_sample(monkeypatch):
    r1 = RuleWithId.create(_make_rule("/b1", "/h"), rule_id="r1")
    r2 = RuleWithId.create(_make_rule("/b2", "/h"), rule_id="r2")
    r3 = RuleWithId.create(_make_rule("/b3", "/h"), rule_id="r3")

    selector = RandomRuleSelector(history=RuleHistory())

    # Make sampling deterministic: pick reversed first k elements
    monkeypatch.setattr("simple_active_refine.rule_selector.random.sample", lambda seq, k: list(seq[:k])[::-1])

    selected, _ = selector.select_rules([r1, r2, r3], k=2, iteration=0)

    assert [r.rule_id for r in selected] == ["r2", "r1"]
