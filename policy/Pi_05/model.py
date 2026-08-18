#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
CameraMode = Literal["head_only", "three_view"]
_CAMERAS_BY_MODE: dict[CameraMode, tuple[str, ...]] = {
    "head_only": ("cam_high",),
    "three_view": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
}


def _validate_camera_mode(value: str) -> CameraMode:
    if value not in _CAMERAS_BY_MODE:
        raise ValueError(f"camera_mode must be one of {tuple(_CAMERAS_BY_MODE)}, got {value!r}")
    return value


def _paired_noise_seed(inference_seed: int, case_meta: dict[str, Any], action_type: str, _env_idx: int) -> int:
    """Derive camera-condition-independent diffusion noise for one formal case."""
    identity = "|".join(
        (
            str(inference_seed),
            str(case_meta.get("task_name", "")),
            str(case_meta.get("seed", "")),
            str(case_meta.get("action_type", action_type)),
        )
    )
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]
        self.camera_mode = _validate_camera_mode(model_cfg.get("camera_mode", "three_view"))
        self.paired_inference_noise = bool(model_cfg.get("paired_inference_noise", False))
        self.inference_seed = int(model_cfg.get("inference_seed", 0))
        self._case_meta: dict[str, Any] = {}
        self._noise_rngs: dict[int, np.random.Generator] = {}

        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy
        self._action_horizon = int(self.policy._model.action_horizon)
        self._action_dim = int(self.policy._model.action_dim)

    def get_model(self, model_cfg: dict[str, Any]):
        train_config_name = model_cfg.get("train_config_name", "pi05_aloha")
        repo_id = model_cfg.get("repo_id", "1118")
        model_root = _resolve_pi05_model_root(model_cfg)

        config = _config.get_config(train_config_name)
        norm_stats = None
        if repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        return _policy_config.create_trained_policy(config, str(model_root), norm_stats=norm_stats)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, camera_mode=self.camera_mode)
            for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        # actions = self.policy.infer(self.observation_window, **kwargs)["actions"]
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            infer_kwargs = dict(kwargs)
            if self.paired_inference_noise and "noise" not in infer_kwargs:
                infer_kwargs["noise"] = self._next_noise(int(env_idx_list[batch_index]))
            actions = self.policy.infer(single_observation, **infer_kwargs)["actions"]
            if self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def prepare_case(self, case_meta=None):
        self._case_meta = dict(case_meta or {})
        self._noise_rngs = {}
        return {
            "camera_mode": self.camera_mode,
            "paired_inference_noise": self.paired_inference_noise,
            "inference_seed": self.inference_seed,
        }

    def _next_noise(self, env_idx: int) -> np.ndarray:
        if env_idx not in self._noise_rngs:
            seed = _paired_noise_seed(self.inference_seed, self._case_meta, self.action_type, env_idx)
            self._noise_rngs[env_idx] = np.random.default_rng(seed)
        return self._noise_rngs[env_idx].standard_normal(
            (self._action_horizon, self._action_dim),
            dtype=np.float32,
        )

    def reset_obsrvationwindows(self):
        self.reset()


def encode_obs(observation, action_type, robot_action_dim_info, camera_mode: CameraMode = "three_view"):
    camera_mode = _validate_camera_mode(camera_mode)
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            camera_name: ensure_chw_uint8(observation["images"][camera_name])
            for camera_name in _CAMERAS_BY_MODE[camera_mode]
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    candidates = {
        "cam_high": ["cam_high", "cam_head", "head_camera", "top_camera"],
        "cam_left_wrist": ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
        "cam_right_wrist": ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
    }
    images = {
        camera_name: ensure_chw_uint8(extract_image(observation, candidates[camera_name]))
        for camera_name in _CAMERAS_BY_MODE[camera_mode]
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    if not obs_list:
        raise ValueError("obs_list must not be empty")
    camera_names = tuple(obs_list[0]["images"])
    if any(tuple(obs["images"]) != camera_names for obs in obs_list[1:]):
        raise ValueError("All observations in a batch must expose the same cameras")
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            camera_name: np.stack([obs["images"][camera_name] for obs in obs_list], axis=0)
            for camera_name in camera_names
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {
            camera_name: image_batch[batch_index]
            for camera_name, image_batch in obs["images"].items()
        },
        "prompt": obs["prompt"][batch_index],
    }


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))
