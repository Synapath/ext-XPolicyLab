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
    if protocol.get("schema") == "tubes-single-insertion-v1":
        from manip_rlt.tubes_episode import run_episode as run_tubes
        return run_tubes(env, client, protocol)
    if protocol["schema"] != "charger-rlt-interaction-v1" or env.num_envs != 1:
        raise ValueError("RLT interaction protocol/single environment required")
    actor_enabled = protocol["actor_enabled"]
    if actor_enabled:
        from manip_rlt.calibration import verify_entry_receipt

        verified = verify_entry_receipt(
            protocol["gate_calibration_path"], protocol["gate_calibration_id"], protocol["gate_config"],
            context=protocol["calibration_context"],
        )
        if verified["identity"] != protocol["gate_calibration_id"]:
            raise ValueError("actor execution requires verified independent entry review")
    if int(os.environ.get("PI05_EXECUTION_HORIZON", "0")) != 10:
        raise ValueError("RLT requires E10")
    if protocol.get("fixed_state"):
        from manip_rlt.fixed_state import run_fixed_state

        return run_fixed_state(env, client, protocol)
    client.call(func_name="reset")
    codec = ActionCodec(protocol["q01"], protocol["q99"], protocol["norm_id"])
    gate = ContactGate(protocol["gate_config"], calibration_id=protocol.get("gate_calibration_id"))
    episode = f"{env.run_id}-s{env.eval_seed}-l{env.env_seeds[0]}"
    record_images = os.environ.get(
        "ROBODOJO_RLT_RECORD_IMAGES", "1" if protocol.get("record_images", True) else "0"
    ) == "1"
    trace = TraceStore(Path(env.save_dir) / ("rlt-" + episode), record_images=record_images)
    oracle = None
    oracle_config = protocol.get("oracle")
    autonomous_case = oracle_config and [int(env.eval_seed), int(env.env_seeds[0])] in oracle_config.get("autonomous_cases", [])
    if oracle_config and not autonomous_case:
        from utils.charger_oracle import ChargerOracle
        oracle = ChargerOracle(env, oracle_config)
    driver = Driver(
        DojoEnvironment(env),
        RemoteProvider(client),
        codec,
        gate,
        RemoteActor(client) if actor_enabled else None,
        RemoteSink(client),
        oracle=oracle,
        handoff_mode=protocol.get("handoff_mode", "chunk-boundary"),
    )
    result = driver.run(episode, trace=trace)
    summary = {k: v for k, v in result.items() if k != "transitions"}
    summary["oracle_allowed"] = bool(oracle_config and not autonomous_case)
    client.call(func_name="rlt_episode_end", obs={**summary, "episode_id": episode})
    summary.update(
        schema="charger-rlt-summary-v1",
        experiment_name=protocol["experiment_name"],
        transition_count=len(result["transitions"]),
        gate_calibration_id=protocol.get("gate_calibration_id"),
    )
    (trace.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
