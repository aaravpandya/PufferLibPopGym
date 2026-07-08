import ctypes

import numpy as np
import pytest


def _skip_if_wrong_env(expected):
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != expected:
        pytest.skip(
            "Build the target env first: "
            f"PYTHON=.venv/bin/python ./build.sh {expected} --cpu"
        )
    return C


def _arrays(vec):
    obs = np.ctypeslib.as_array(
        (ctypes.c_ubyte * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
    ).reshape(vec.total_agents, vec.obs_size)
    rewards = np.ctypeslib.as_array(
        (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
    )
    terminals = np.ctypeslib.as_array(
        (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
    )
    return obs, rewards, terminals


@pytest.mark.parametrize(
    ("env_name", "num_values", "num_actions", "episode_length"),
    [
        ("popgym_count_recall_medium", 4, 27, 103),
        ("popgym_count_recall_hard", 13, 17, 207),
    ],
)
def test_count_recall_variant_metadata_and_correct_count(
    env_name, num_values, num_actions, episode_length
):
    C = _skip_if_wrong_env(env_name)
    args = {"vec": {"total_agents": 1, "num_buffers": 1, "num_threads": 1}, "env": {}}
    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [num_actions]

        obs, rewards, terminals = _arrays(vec)
        assert np.all(obs < num_values)

        counts = [0] * num_values
        counts[int(obs[0, 0])] += 1
        prev_query = int(obs[0, 1])
        action = np.array([[counts[prev_query]]], dtype=np.float32)
        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(1.0 / episode_length)
        assert terminals[0] == 0.0
        assert np.all(obs < num_values)
    finally:
        vec.close()
