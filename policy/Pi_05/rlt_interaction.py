"""Single-env RLT driver bridge. Imports the learner-free adapter only when enabled."""

import numpy as np

from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

DIMS = {"arm_dim": [6, 6], "ee_dim": [1, 1]}


class DojoEnvironment:
    def __init__(self, env):
        from utils import rlt_interaction

        self.env, self.native = env, rlt_interaction

    def _convert(self, frame):
        frame["proprio"] = pack_robot_state(frame["observation"], "joint", DIMS, source_type="obs").astype(np.float32)
        frame["applied"] = (
            None
            if frame["applied_dict"] is None
            else pack_robot_state({"state": frame["applied_dict"]}, "joint", DIMS, source_type="obs")
        )
        return frame

    def observe(self):
        return self._convert(self.native.read_frame(self.env))

    def step(self, command):
        action = unpack_robot_state(np.asarray(command)[None], "joint", DIMS, source_type="obs")[0]
        return self._convert(self.native.step(self.env, action))


class RemoteProvider:
    def __init__(self, client):
        self.client = client

    def infer(self, observation, *, request_id):
        self.client.call(func_name="update_obs", obs=observation)
        return self.client.call(func_name="rlt_features", obs={"request_id": request_id})


class RemoteActor:
    def __init__(self, client):
        self.client = client

    def act(self, z, proprio, reference):
        result = self.client.call(func_name="rlt_action", obs={"z": z, "proprio": proprio, "reference": reference})
        return np.asarray(result["action"]), int(result["actor_version"])


class RemoteSink:
    def __init__(self, client):
        self.client = client

    def append(self, record):
        self.client.call(func_name="rlt_transition", obs=record)


def run_episode(env, client):
    import json
    import os
    from pathlib import Path

    from manip_rlt.driver import ActionCodec, Driver, TraceStore
    from manip_rlt.gate import ContactGate

    protocol = json.loads(Path(os.environ["ROBODOJO_RLT_PROTOCOL"]).read_text())
    if protocol["schema"] != "charger-rlt-interaction-v1" or env.num_envs != 1:
        raise ValueError("RLT interaction protocol/single environment required")
    actor_enabled = protocol["actor_enabled"]
    if actor_enabled and not protocol.get("gate_calibration_id"):
        raise ValueError("actor execution requires independent gate calibration")
    if int(os.environ.get("PI05_EXECUTION_HORIZON", "0")) != 10:
        raise ValueError("RLT requires E10")
    client.call(func_name="reset")
    codec = ActionCodec(protocol["q01"], protocol["q99"], protocol["norm_id"])
    gate = ContactGate(protocol["gate_config"], calibration_id=protocol.get("gate_calibration_id"))
    episode = f"{env.run_id}-s{env.eval_seed}-l{env.env_seeds[0]}"
    trace = TraceStore(Path(env.save_dir) / ("rlt-" + episode))
    driver = Driver(
        DojoEnvironment(env),
        RemoteProvider(client),
        codec,
        gate,
        RemoteActor(client) if actor_enabled else None,
        RemoteSink(client),
    )
    result = driver.run(episode, trace=trace)
    summary = {k: v for k, v in result.items() if k != "transitions"}
    summary.update(
        schema="charger-rlt-summary-v1",
        experiment_name=protocol["experiment_name"],
        transition_count=len(result["transitions"]),
        gate_calibration_id=protocol.get("gate_calibration_id"),
    )
    (trace.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
