import pytest

from simple_active_refine.amie import AmieRule, AmieRules, TriplePattern
from simple_active_refine.arm import Arm, ArmWithId
from simple_active_refine.arm_pipeline import ArmDrivenKGRefinementPipeline, ArmPipelineConfig


def test_arm_pipeline_witness_weighting_by_relation_priors(tmp_path):
    kg_train = [("a", "p1", "b")]
    candidates = [
        ("a", "p1", "b"),
        ("b", "p2", "c"),
        ("a", "p1", "x"),
        ("x", "p2", "c"),
    ]
    targets = [("a", "r*", "c")]

    rule = AmieRule(
        head=TriplePattern("?x", "r*", "?y"),
        body=[TriplePattern("?x", "p1", "?z"), TriplePattern("?z", "p2", "?y")],
        support=None,
        std_conf=None,
        pca_conf=None,
        head_coverage=None,
        body_size=None,
        pca_body_size=None,
        raw="test",
    )
    rules = AmieRules([rule])
    arms = [Arm(arm_type="set", rule_keys=[str(rule)], metadata={"kind": "singleton"})]

    cfg = ArmPipelineConfig(
        base_output_path=str(tmp_path / "out"),
        n_iter=1,
        k_sel=1,
        n_targets_per_arm=1,
        selector_strategy="ucb",
        witness_weight=1.0,
        evidence_weight=1.0,
    )

    # Rule weight W(h) = X(p1) * X(p2) = 0.5 * 0.2 = 0.1
    # Raw witness count is 2, so weighted witness score is 0.2.
    # Accepted evidence count is 3 (everything except (a,p1,b) which is already in KG).
    expected_reward = 0.2 + 3.0

    pipe = ArmDrivenKGRefinementPipeline(
        config=cfg,
        arm_pool=[ArmWithId.create(arms[0])],
        rule_pool=rules,
        kg_train_triples=kg_train,
        target_triples=targets,
        candidate_triples=candidates,
        relation_priors={"p1": 0.5, "p2": 0.2},
    )
    pipe.run()

    assert len(pipe.history.records) == 1
    rec = pipe.history.records[0]
    assert rec.witness_by_target[targets[0]] == 2
    assert rec.reward == pytest.approx(expected_reward)


def test_arm_pipeline_smoke(tmp_path):
    # Tiny KG and candidates
    kg_train = [("a", "p1", "b")]
    candidates = [
        ("a", "p1", "b"),
        ("b", "p2", "c"),
        ("a", "p1", "x"),
        ("x", "p2", "c"),
        # Not part of the rule body, but should be added as incident context
        # once 'x' is introduced by accepted evidence.
        ("x", "p3", "y"),
    ]
    targets = [("a", "r*", "c")]

    # Rule: p1 then p2 implies r*
    rule = AmieRule(
        head=TriplePattern("?x", "r*", "?y"),
        body=[TriplePattern("?x", "p1", "?z"), TriplePattern("?z", "p2", "?y")],
        support=None,
        std_conf=None,
        pca_conf=None,
        head_coverage=None,
        body_size=None,
        pca_body_size=None,
        raw="test",
    )
    rules = AmieRules([rule])

    arms = [Arm(arm_type="set", rule_keys=[str(rule)], metadata={"kind": "singleton"})]

    cfg = ArmPipelineConfig(
        base_output_path=str(tmp_path / "out"),
        n_iter=1,
        k_sel=1,
        n_targets_per_arm=1,
        selector_strategy="ucb",
    )

    pipe = ArmDrivenKGRefinementPipeline(
        config=cfg,
        arm_pool=[ArmWithId.create(arms[0])],
        rule_pool=rules,
        kg_train_triples=kg_train,
        target_triples=targets,
        candidate_triples=candidates,
    )

    pipe.run()

    iter_dir = tmp_path / "out" / "iter_1"
    assert (iter_dir / "selected_arms.json").exists()
    assert (iter_dir / "accepted_evidence_triples.tsv").exists()
    assert (iter_dir / "accepted_incident_triples.tsv").exists()
    assert (iter_dir / "accepted_added_triples.tsv").exists()
    assert (iter_dir / "pending_hypothesis_triples.tsv").exists()
    assert (iter_dir / "arm_history.pkl").exists()
    assert (iter_dir / "arm_history.json").exists()
    assert (iter_dir / "diagnostics.json").exists()

    # Verify the incident triple was added.
    added = (iter_dir / "accepted_added_triples.tsv").read_text(encoding="utf-8")
    assert "x\tp3\ty" in added
