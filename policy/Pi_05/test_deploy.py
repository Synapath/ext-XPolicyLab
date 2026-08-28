import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("deploy.py")
SPEC = importlib.util.spec_from_file_location("pi05_deploy", MODULE_PATH)
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


class FakeSingleEnv:
    def __init__(self, episode_steps):
        self.episode_steps = episode_steps
        self.actions = []

    def is_episode_end(self):
        return len(self.actions) >= self.episode_steps

    def get_obs(self):
        return {"step": len(self.actions)}

    def take_action(self, action):
        self.actions.append(action)


class FakeBatchEnv:
    def __init__(self, episode_steps):
        self.episode_steps = episode_steps
        self.step = 0
        self.actions = []

    def is_episode_end(self):
        return self.step >= self.episode_steps

    def get_running_env_idx_list(self):
        return [] if self.is_episode_end() else [0, 1]

    def get_obs_batch(self, env_idx_list):
        return [{"env": env_idx, "step": self.step} for env_idx in env_idx_list]

    def take_action_batch(self, actions, env_idx_list):
        self.actions.append((tuple(env_idx_list), tuple(actions)))
        self.step += 1


class FakeClient:
    def __init__(self, chunk_size=50):
        self.chunk_size = chunk_size
        self.calls = []

    def call(self, func_name, **kwargs):
        self.calls.append((func_name, kwargs))
        if func_name == "get_action":
            return list(range(self.chunk_size))
        if func_name == "get_action_batch":
            return [list(range(self.chunk_size)) for _ in kwargs["obs"]]
        if func_name not in {"reset", "update_obs", "update_obs_batch"}:
            raise AssertionError(func_name)
        return None


class DeployControllerTest(unittest.TestCase):
    def _run(self, execution_horizon):
        env = FakeSingleEnv(episode_steps=100)
        client = FakeClient()
        with mock.patch.dict(os.environ, {"PI05_EXECUTION_HORIZON": str(execution_horizon)}):
            deploy.eval_one_episode(env, client)
        get_action_calls = [call for call in client.calls if call[0] == "get_action"]
        update_obs_calls = [call for call in client.calls if call[0] == "update_obs"]
        return env, get_action_calls, update_obs_calls

    def test_full_h50_executes_whole_chunk(self):
        env, get_action_calls, update_obs_calls = self._run(50)
        self.assertEqual(env.actions, list(range(50)) * 2)
        self.assertEqual(len(get_action_calls), 2)
        self.assertEqual(len(update_obs_calls), 100)

    def test_sync20_replans_after_twenty_actions(self):
        env, get_action_calls, update_obs_calls = self._run(20)
        self.assertEqual(env.actions, list(range(20)) * 5)
        self.assertEqual(len(get_action_calls), 5)
        self.assertEqual(len(update_obs_calls), 100)

    def test_sync16_preserves_prefix_and_termination(self):
        env, calls, observations = self._run(16)
        self.assertEqual(env.actions, list(range(16)) * 6 + list(range(4)))
        self.assertEqual(len(calls), 7)
        self.assertEqual(len(observations), 100)
        self.assertEqual([x[1]["obs"]["step"] for x in observations], list(range(100)))

    def test_sync16_batch_removes_finished_environment(self):
        class UnevenEnv:
            def __init__(self):
                self.actions = {0: [], 1: []}

            def is_episode_end(self):
                return not self.get_running_env_idx_list()

            def get_running_env_idx_list(self):
                return [i for i, stop in ((0, 5), (1, 35)) if len(self.actions[i]) < stop]

            def get_obs_batch(self, ids):
                return [{"env": i, "step": len(self.actions[i])} for i in ids]

            def take_action_batch(self, actions, ids):
                for i, a in zip(ids, actions):
                    self.actions[i].append(a)

        env, client = UnevenEnv(), FakeClient()
        with mock.patch.dict(os.environ, {"PI05_EXECUTION_HORIZON": "16"}):
            deploy.eval_one_episode_batch(env, client)
        self.assertEqual(env.actions[0], list(range(5)))
        self.assertEqual(env.actions[1], list(range(16)) * 2 + list(range(3)))
        requests = [x[1]["obs"] for x in client.calls if x[0] == "get_action_batch"]
        self.assertEqual(requests, [[0, 1], [1], [1]])

    def test_sync16_wire_sequence_is_deterministic(self):
        first = self._run(16)
        second = self._run(16)
        self.assertEqual(first[0].actions, second[0].actions)
        self.assertEqual(first[1:], second[1:])

    def test_arx_shape_and_finite_boundary(self):
        import numpy as np

        self.assertEqual(deploy.validate_arx_prediction(np.zeros((50, 14))).shape, (50, 14))
        for bad in (np.zeros((16, 14)), np.zeros((50, 32)), np.full((50, 14), np.nan)):
            with self.assertRaisesRegex(ValueError, "finite"):
                deploy.validate_arx_prediction(bad)

    def test_batch_sync20_replans_after_twenty_actions(self):
        env = FakeBatchEnv(episode_steps=100)
        client = FakeClient()
        with mock.patch.dict(os.environ, {"PI05_EXECUTION_HORIZON": "20"}):
            deploy.eval_one_episode_batch(env, client)
        get_action_calls = [call for call in client.calls if call[0] == "get_action_batch"]
        self.assertEqual(len(get_action_calls), 5)
        self.assertEqual(len(env.actions), 100)
        self.assertTrue(all(env_idx_list == (0, 1) for env_idx_list, _ in env.actions))

    def test_rejects_unregistered_execution_horizon(self):
        env = FakeSingleEnv(episode_steps=1)
        client = FakeClient()
        with mock.patch.dict(os.environ, {"PI05_EXECUTION_HORIZON": "10"}):
            with self.assertRaisesRegex(ValueError, "must be one of"):
                deploy.eval_one_episode(env, client)

    def test_rejects_non_h50_prediction(self):
        env = FakeSingleEnv(episode_steps=1)
        client = FakeClient(chunk_size=49)
        with mock.patch.dict(os.environ, {"PI05_EXECUTION_HORIZON": "20"}):
            with self.assertRaisesRegex(ValueError, "exactly H50"):
                deploy.eval_one_episode(env, client)


if __name__ == "__main__":
    unittest.main()
