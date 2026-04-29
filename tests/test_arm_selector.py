import pytest

from simple_active_refine.arm import Arm, ArmWithId
from simple_active_refine.arm_history import ArmEvaluationRecord, ArmHistory
from simple_active_refine.arm_selector import (
    ArmSelectionPolicy,
    EpsilonGreedyArmSelector,
    LLMPolicyArmSelector,
    UCBArmSelector,
)


def _add_history(history: ArmHistory, arm_with_id: ArmWithId, rewards: list[float]) -> None:
    rec = ArmEvaluationRecord(
        iteration=1,
        arm_id=arm_with_id.arm_id,
        arm=arm_with_id.arm,
        target_triples=[("h", "r", "t")],
        added_triples=[("a", "b", "c")],
        reward=sum(rewards) / len(rewards),
        diagnostics={},
    )
    history.add_record(rec)


def test_llm_policy_uses_history_and_mocked_llm(monkeypatch):
    a1 = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk1"], metadata={}), arm_id="a1")
    a2 = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk2"], metadata={}), arm_id="a2")
    a3 = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk3"], metadata={}), arm_id="a3")

    history = ArmHistory()
    _add_history(history, a1, [0.1])

    selector = LLMPolicyArmSelector(history=history, chat_model="gpt-4o", temperature=0.0)

    class _DummyLLM:
        def invoke(self, prompt):
            return ArmSelectionPolicy(
                reasoning="mock",
                policy_text="mock policy",
                selected_arm_ids=["a2", "a1"],
                rationale_per_arm={"a2": "explore", "a1": "exploit"},
            )

    monkeypatch.setattr(selector, "structured_llm", _DummyLLM())

    selected, policy = selector.select_arms([a1, a2, a3], k=2, iteration=2)

    assert [a.arm_id for a in selected] == ["a2", "a1"]
    assert policy == "mock policy"


def test_llm_policy_iteration0_respects_pool_order():
    a1 = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk1"], metadata={}), arm_id="a1")
    a2 = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk2"], metadata={}), arm_id="a2")

    selector = LLMPolicyArmSelector(history=ArmHistory(), chat_model="gpt-4o", temperature=0.0)
    selected, _ = selector.select_arms([a1, a2], k=1, iteration=0)

    assert [a.arm_id for a in selected] == ["a1"]


def test_ucb_prefers_untried_arms():
    tried = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk1"], metadata={}), arm_id="tried")
    new = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk2"], metadata={}), arm_id="new")

    history = ArmHistory()
    _add_history(history, tried, [0.1])

    selector = UCBArmSelector(history=history, exploration_param=1.0)
    selected, _ = selector.select_arms([tried, new], k=1, iteration=3)

    assert [a.arm_id for a in selected] == ["new"]


def test_epsilon_greedy_exploit_chooses_best_mean():
    low = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk1"], metadata={}), arm_id="low")
    high = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk2"], metadata={}), arm_id="high")

    history = ArmHistory()
    _add_history(history, low, [0.05])
    _add_history(history, high, [0.5])

    selector = EpsilonGreedyArmSelector(history=history, epsilon=0.0)
    selected, _ = selector.select_arms([low, high], k=1, iteration=2)

    assert [a.arm_id for a in selected] == ["high"]


def test_epsilon_greedy_explore_uses_random(monkeypatch):
    a1 = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk1"], metadata={}), arm_id="a1")
    a2 = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk2"], metadata={}), arm_id="a2")

    selector = EpsilonGreedyArmSelector(history=ArmHistory(), epsilon=1.0)

    monkeypatch.setattr("simple_active_refine.arm_selector.random.random", lambda: 0.0)
    monkeypatch.setattr("simple_active_refine.arm_selector.random.choice", lambda seq: seq[0])

    selected, _ = selector.select_arms([a1, a2], k=1, iteration=1)
    assert [a.arm_id for a in selected] == ["a1"]


def test_create_llm_policy_does_not_call_api_in_tests():
    # Just sanity-check construction; tests must mock structured_llm for invocation.
    selector = LLMPolicyArmSelector(history=ArmHistory(), chat_model="gpt-4o", temperature=0.0)
    assert selector is not None


def test_llm_policy_prompt_includes_entity_and_relation_texts():
    # Ensure the prompt actually contains semantic context (entity2text / relation2text)
    # so the LLM can reason about meaning beyond IDs.
    history = ArmHistory()

    arm = ArmWithId.create(Arm(arm_type="set", rule_keys=["rk1"], metadata={}), arm_id="a1")
    rec = ArmEvaluationRecord(
        iteration=1,
        arm_id=arm.arm_id,
        arm=arm.arm,
        target_triples=[("e_alice", "/people/person/nationality", "e_japan")],
        added_triples=[("e_alice", "/people/person/place_of_birth", "e_tokyo")],
        reward=0.1,
        diagnostics={"targets_total": 1, "targets_with_witness": 1, "target_coverage": 1.0},
    )
    history.add_record(rec)

    selector = LLMPolicyArmSelector(
        history=history,
        chat_model="gpt-4o",
        temperature=0.0,
        target_predicates=["/people/person/nationality"],
        relation_texts={
            "/people/person/nationality": "nationality of a person",
            "/people/person/place_of_birth": "place where a person was born",
        },
        entity_texts={
            "e_alice": "Alice",
            "e_japan": "Japan",
            "e_tokyo": "Tokyo",
        },
    )

    prompt = selector._create_selection_prompt([arm], k=1, iteration=2)
    assert "Alice" in prompt
    assert "Japan" in prompt
    assert "nationality of a person" in prompt
    assert "Semantic Grounding" in prompt
