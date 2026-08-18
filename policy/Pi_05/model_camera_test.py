import numpy as np
import pytest

from XPolicyLab.policy.Pi_05.model import _paired_noise_seed, encode_obs, slice_stacked_obs, stack_obs


def _observation() -> dict:
    return {
        "state": np.arange(14, dtype=np.float32),
        "images": {
            "cam_high": np.full((3, 8, 8), 1, dtype=np.uint8),
            "cam_left_wrist": np.full((3, 8, 8), 2, dtype=np.uint8),
            "cam_right_wrist": np.full((3, 8, 8), 3, dtype=np.uint8),
        },
        "instruction": "place the object",
    }


def test_head_only_does_not_require_or_forward_wrist_images() -> None:
    observation = _observation()
    observation["images"].pop("cam_left_wrist")
    observation["images"].pop("cam_right_wrist")

    encoded = encode_obs(observation, "joint", None, camera_mode="head_only")

    assert tuple(encoded["images"]) == ("cam_high",)
    assert encoded["prompt"] == observation["instruction"]
    np.testing.assert_array_equal(encoded["state"], observation["state"])


def test_three_view_batch_round_trip_preserves_all_cameras() -> None:
    encoded = encode_obs(_observation(), "joint", None, camera_mode="three_view")
    stacked = stack_obs([encoded, encoded])
    sliced = slice_stacked_obs(stacked, 1)

    assert tuple(sliced["images"]) == ("cam_high", "cam_left_wrist", "cam_right_wrist")
    for camera_name in sliced["images"]:
        np.testing.assert_array_equal(sliced["images"][camera_name], encoded["images"][camera_name])


def test_invalid_camera_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="camera_mode"):
        encode_obs(_observation(), "joint", None, camera_mode="third_person")


def test_paired_noise_seed_depends_on_case_but_not_camera_condition() -> None:
    case = {"task_name": "press_stapler", "seed": 920001, "action_type": "joint"}

    head_seed = _paired_noise_seed(7, case, "joint", 0)
    three_view_seed = _paired_noise_seed(7, case, "joint", 0)
    next_case_seed = _paired_noise_seed(7, {**case, "seed": 920002}, "joint", 0)

    assert head_seed == three_view_seed
    assert head_seed != next_case_seed
