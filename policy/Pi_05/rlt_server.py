"""H32 ARX policy serving with isolated observation/noise state per trial.

Runs on the training/serving host; does not modify the deployed simulator.
"""

import argparse
import asyncio
import contextvars
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from openpi.policies import policy_config
from openpi.training import config
from XPolicyLab.policy.Pi_05.model import encode_obs
from XPolicyLab.utils.process_data import unpack_robot_state


_trial = contextvars.ContextVar("pi05_trial")
_DIMENSIONS = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


class TrialPolicyServer(PolicyServer):
    async def _execute_frame(self, frame):
        token = _trial.set((frame.evaluation_id, frame.trial_id))
        try:
            return await super()._execute_frame(frame)
        finally:
            _trial.reset(token)


class RltModel:
    def __init__(self, policy, receipt_path, checkpoint_ref, *, horizon=32, seed=0,
                 instruction="Plug the charger into the power strip."):
        self.policy, self.receipt_path, self.checkpoint_ref = policy, Path(receipt_path), checkpoint_ref
        self.horizon, self.seed = horizon, seed
        self.instruction = instruction
        self.sessions = {}

    def reset(self):
        key = _trial.get()
        if key not in self.sessions and len(self.sessions) >= 128:
            raise RuntimeError("trial capacity reached; restart serving between evaluation groups")
        self.sessions[key] = {"observations": {}, "counts": {}, "indices": []}

    def _session(self):
        if _trial.get() not in self.sessions:
            raise ValueError("reset is required before observations/actions")
        return self.sessions[_trial.get()]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, observations):
        session = self._session()
        indices = []
        for i, obs in enumerate(observations):
            index = int(obs.get("env_idx", i))
            if index in indices:
                raise ValueError("duplicate environment identity")
            encoded = encode_obs(obs, "joint", _DIMENSIONS)
            if encoded["state"].shape != (14,) or not np.isfinite(encoded["state"]).all():
                raise ValueError("expected finite ARX 14D state")
            if encoded["prompt"] != self.instruction:
                raise ValueError("task prompt mismatch")
            session["observations"][index] = encoded
            indices.append(index)
        session["indices"] = indices

    def get_action(self):
        session = self._session()
        if len(session["indices"]) != 1:
            raise ValueError("single action requested for a batch")
        return self.get_action_batch(session["indices"])[0]

    def get_action_batch(self, env_idx_list=None):
        session = self._session()
        ids = session["indices"] if env_idx_list is None else env_idx_list
        result = []
        for index in ids:
            if index not in session["indices"]:
                raise ValueError("action requested for stale/inactive observation")
            observation = session["observations"][index]
            count = session["counts"].get(index, 0)
            key = jax.random.fold_in(jax.random.fold_in(jax.random.key(self.seed), index), count)
            noise = np.asarray(jax.random.normal(key, (self.horizon, 32)))
            actions = np.asarray(self.policy.infer(observation, noise=noise)["actions"])
            if actions.shape != (self.horizon, 14) or not np.isfinite(actions).all():
                raise ValueError("non-finite or wrong-shape ARX action chunk")
            session["counts"][index] = count + 1
            with self.receipt_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "evaluation_trial": _trial.get(),
                            "env_idx": index,
                            "inference_index": count,
                            "policy_seed": self.seed,
                            "horizon": self.horizon,
                            "checkpoint_ref": self.checkpoint_ref,
                            "state": observation["state"].tolist(),
                            "actions": actions.tolist(),
                            "noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
                            "cameras": {
                                k: {"shape": list(v.shape), "sha256": hashlib.sha256(v.tobytes()).hexdigest()}
                                for k, v in observation["images"].items()
                            },
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            result.append(unpack_robot_state(actions, "joint", _DIMENSIONS, source_type="obs"))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-ref", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19032)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("serving receipt already exists")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    policy = policy_config.create_trained_policy(config.get_config("pi05_rlt_charger_r1_0a"), args.checkpoint)
    server = TrialPolicyServer(
        RltModel(policy, args.receipt, args.checkpoint_ref),
        PolicyServerConfig(host=args.host, port=args.port, ws_ping_timeout_s=None),
    )
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
