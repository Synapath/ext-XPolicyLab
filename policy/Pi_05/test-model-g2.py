"""CPU integration of G2's native 14-D boundary; no simulator/model download."""

import numpy as np
import pytest

from XPolicyLab.policy.Pi_05 import model
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state


def test_three_camera_state_language_mapping():
    dims = get_robot_action_dim_info("arx_x5")
    state = np.arange(14, dtype=np.float32)
    cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    raw = {
        "state": unpack_robot_state(state, "joint", dims),
        "vision": {c: {"color": np.full((8, 12, 3), i + 1, dtype=np.uint8)} for i, c in enumerate(cameras)},
        "instruction": "Fold the clothes neatly.",
    }
    encoded = model.encode_obs(raw, "joint", dims)
    np.testing.assert_array_equal(encoded["state"], state)
    assert encoded["prompt"] == raw["instruction"]
    for i, camera in enumerate(cameras):
        assert encoded["images"][camera].shape == (3, 8, 12)
        assert np.all(encoded["images"][camera] == i + 1)


@pytest.mark.parametrize("bad", [False, True])
def test_model_checks_g2_action_before_native_unpack(monkeypatch, bad):
    class Policy:
        def infer(self, observation, **kwargs):
            actions = np.arange(50 * 14, dtype=np.float32).reshape(50, 14)
            if bad:
                actions[0, 6] = np.nan
            return {"actions": actions}

    monkeypatch.setattr(model.Model, "get_model", lambda *args, **kwargs: Policy())
    instance = model.Model({"task_name": "fold_clothes", "env_cfg_type": "arx_x5",
                            "train_config_name": "pi05_g2_rbdj_three_task_s0_strict"})
    instance.update_obs({"state": np.zeros(14, dtype=np.float32),
                         "images": {c: np.zeros((3, 8, 12), dtype=np.uint8)
                                    for c in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
                         "instruction": "Fold the clothes neatly."})
    if bad:
        with pytest.raises(ValueError, match="finite"):
            instance.get_action()
    else:
        result = instance.get_action()
        assert len(result) == 50
        packed = np.stack([pack_robot_state({"state": row}, "joint", instance.robot_action_dim_info)
                           for row in result])
        np.testing.assert_array_equal(packed, np.arange(700, dtype=np.float32).reshape(50, 14))
