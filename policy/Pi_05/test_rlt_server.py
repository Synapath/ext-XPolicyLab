import numpy as np
import pytest

from XPolicyLab.policy.Pi_05.rlt_server import RltModel, _trial


class FakePolicy:
    def infer(self, obs, *, noise):
        # Preserve observation identity in all actions; record stochastic input.
        self.noise = noise.copy()
        return {"actions": np.broadcast_to(obs["state"], (32, 14)).copy()}


def observation(value, index=0):
    return {
        "env_idx": index,
        "state": np.full(14, value, np.float32),
        "images": {name: np.zeros((3, 8, 8), np.uint8) for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
        "instruction": "Plug the charger into the power strip.",
    }


def test_interleaved_trial_observations_and_noise_are_isolated(tmp_path):
    policy = FakePolicy()
    model = RltModel(policy, tmp_path / "receipt.jsonl", "test-only")
    token = _trial.set(("eval-a", "trial-a"))
    try:
        model.reset()
        model.update_obs(observation(0.1))
        _trial.set(("eval-b", "trial-b"))
        model.reset()
        model.update_obs(observation(0.2))
        b = model.get_action()
        first_noise = policy.noise.copy()
        _trial.set(("eval-a", "trial-a"))
        a = model.get_action()
        np.testing.assert_array_equal(policy.noise, first_noise)
        np.testing.assert_allclose(a[0]["left_arm_joint_state"], 0.1)
        np.testing.assert_allclose(b[0]["left_arm_joint_state"], 0.2)
        assert len(a) == len(b) == 32
        model.get_action()
        assert not np.array_equal(policy.noise, first_noise)
        model.reset()
        model.update_obs(observation(0.1))
        model.get_action()
        np.testing.assert_array_equal(policy.noise, first_noise)
    finally:
        _trial.reset(token)


def test_rejects_missing_reset_and_wrong_prompt(tmp_path):
    model = RltModel(FakePolicy(), tmp_path / "receipt.jsonl", "test-only")
    token = _trial.set(("eval", "trial"))
    try:
        with pytest.raises(ValueError, match="reset"):
            model.update_obs(observation(0.1))
        model.reset()
        obs = observation(0.1)
        obs["instruction"] = "Stack bowls."
        with pytest.raises(ValueError, match="prompt"):
            model.update_obs(obs)
    finally:
        _trial.reset(token)


def test_protocol_dispatch_propagates_trial_context_to_threads(tmp_path):
    import asyncio
    from client_server.ws.protocol.messages import MessageType
    from client_server.ws.protocol.schemas import Frame
    from XPolicyLab.policy.Pi_05.rlt_server import TrialPolicyServer

    server = TrialPolicyServer(RltModel(FakePolicy(), tmp_path / "wire.jsonl", "test-only"))
    counter = 0

    async def call(trial, message_type, payload=None):
        nonlocal counter
        counter += 1
        response = await server.process_frame(
            Frame(
                message_type=message_type,
                request_id=str(counter),
                evaluation_id="eval",
                trial_id=trial,
                payload=payload or {},
            )
        )
        assert response.message_type != MessageType.ERROR
        return response.payload

    async def run():
        await asyncio.gather(call("a", MessageType.RESET), call("b", MessageType.RESET))
        await asyncio.gather(
            call("a", MessageType.CALL, {"func_name": "update_obs", "obs": observation(0.1)}),
            call("b", MessageType.CALL, {"func_name": "update_obs", "obs": observation(0.2)}),
        )
        a, b = await asyncio.gather(
            call("a", MessageType.CALL, {"func_name": "get_action"}),
            call("b", MessageType.CALL, {"func_name": "get_action"}),
        )
        np.testing.assert_allclose(a["result"][0]["left_arm_joint_state"], 0.1)
        np.testing.assert_allclose(b["result"][0]["left_arm_joint_state"], 0.2)

    asyncio.run(run())
