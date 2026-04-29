import pickle

from simple_active_refine.arm import Arm, ArmWithId
from simple_active_refine.arm_builder import load_arm_pool_with_ids, load_arms_json, load_arms_pickle, save_arms_json


def test_load_arms_json_roundtrip(tmp_path):
    arms = [
        Arm(arm_type="set", rule_keys=["r1"], metadata={"kind": "singleton"}),
        Arm(arm_type="set", rule_keys=["r2", "r3"], metadata={"kind": "pair", "cooc": 0.5}),
    ]
    p = tmp_path / "initial_arms.json"
    save_arms_json(arms, str(p))

    loaded = load_arms_json(p)
    assert loaded == arms


def test_load_arms_pickle_roundtrip(tmp_path):
    arms = [Arm(arm_type="set", rule_keys=["r1"], metadata={})]
    p = tmp_path / "initial_arms.pkl"
    with open(p, "wb") as f:
        pickle.dump(arms, f)

    loaded = load_arms_pickle(p)
    assert loaded == arms


def test_load_arm_pool_with_ids_deterministic(tmp_path):
    arms = [Arm(arm_type="set", rule_keys=["r1"], metadata={})]
    p = tmp_path / "initial_arms.json"
    save_arms_json(arms, str(p))

    pool1 = load_arm_pool_with_ids(p)
    pool2 = load_arm_pool_with_ids(p)
    assert len(pool1) == 1
    assert isinstance(pool1[0], ArmWithId)
    assert pool1[0].arm_id == pool2[0].arm_id
