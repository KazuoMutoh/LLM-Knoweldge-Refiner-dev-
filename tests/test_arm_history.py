import json

from simple_active_refine.arm import Arm, ArmWithId
from simple_active_refine.arm_history import ArmEvaluationRecord, ArmHistory


def test_arm_with_id_is_deterministic_for_set_arms():
    arm1 = Arm(arm_type="set", rule_keys=["r1", "r2"], metadata={"kind": "pair"})
    arm2 = Arm(arm_type="set", rule_keys=["r2", "r1"], metadata={"kind": "pair"})

    a1 = ArmWithId.create(arm1)
    a2 = ArmWithId.create(arm2)

    assert arm1.key() == arm2.key()
    assert a1.arm_id == a2.arm_id


def test_arm_with_id_differs_for_sequence_order():
    arm1 = Arm(arm_type="sequence", rule_keys=["r1", "r2"], metadata={})
    arm2 = Arm(arm_type="sequence", rule_keys=["r2", "r1"], metadata={})

    a1 = ArmWithId.create(arm1)
    a2 = ArmWithId.create(arm2)

    assert arm1.key() != arm2.key()
    assert a1.arm_id != a2.arm_id


def test_arm_history_save_load_and_json(tmp_path):
    arm = Arm(arm_type="set", rule_keys=["r1"], metadata={"kind": "singleton"})
    arm_with_id = ArmWithId.create(arm)

    history = ArmHistory()
    rec = ArmEvaluationRecord(
        iteration=1,
        arm_id=arm_with_id.arm_id,
        arm=arm,
        target_triples=[("h", "r", "t")],
        added_triples=[("h2", "p2", "t2")],
        reward=0.5,
        diagnostics={"n_witness": 1.0},
    )
    history.add_record(rec)

    pkl_path = tmp_path / "arm_history.pkl"
    json_path = tmp_path / "arm_history.json"

    history.save(str(pkl_path))
    history.save_json(str(json_path))

    loaded = ArmHistory.load(str(pkl_path))
    assert len(loaded.records) == 1
    assert loaded.get_records_for_arm(arm_with_id.arm_id)[0].reward == 0.5

    stat = loaded.get_arm_statistics(arm_with_id.arm_id)
    assert stat is not None
    assert stat.total_iterations == 1
    assert stat.total_triples_added == 1

    # JSON is a list of records
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["arm_id"] == arm_with_id.arm_id
