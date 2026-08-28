import os


PREDICTION_HORIZON = 50
ALLOWED_EXECUTION_HORIZONS = frozenset({16, 20, 50})


def _execution_horizon():
    raw_value = os.environ.get("PI05_EXECUTION_HORIZON", str(PREDICTION_HORIZON))
    try:
        execution_horizon = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"PI05_EXECUTION_HORIZON must be an integer, got {raw_value!r}") from exc

    if execution_horizon not in ALLOWED_EXECUTION_HORIZONS:
        allowed = ", ".join(str(value) for value in sorted(ALLOWED_EXECUTION_HORIZONS))
        raise ValueError(f"PI05_EXECUTION_HORIZON must be one of {{{allowed}}}, got {execution_horizon}")
    return execution_horizon


def _validate_action_chunk(actions):
    chunk_size = len(actions)
    if chunk_size != PREDICTION_HORIZON:
        raise ValueError(
            f"Pi_05 must predict exactly H{PREDICTION_HORIZON}, got action chunk length {chunk_size}"
        )


def validate_arx_prediction(actions):
    """G2 wire boundary before unpacking into native left/right arm dictionaries."""
    import numpy as np

    actions = np.asarray(actions)
    if actions.shape != (50, 14) or not np.isfinite(actions).all():
        raise ValueError(f"G2 requires finite [50,14] ARX actions, got {actions.shape}")
    return actions


def eval_one_episode(TASK_ENV, model_client):
    execution_horizon = _execution_horizon()
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        _validate_action_chunk(actions)

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == execution_horizon:
                break

            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)


def eval_one_episode_batch(TASK_ENV, model_client):
    execution_horizon = _execution_horizon()
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        if len(actions) != len(env_idx_list):
            raise ValueError(
                f"Pi_05 returned {len(actions)} action chunks for {len(env_idx_list)} active environments"
            )
        for env_actions in actions:
            _validate_action_chunk(env_actions)

        for action_idx in range(execution_horizon):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)

            if TASK_ENV.is_episode_end() or action_idx + 1 == execution_horizon:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
