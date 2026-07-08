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


@pytest.mark.parametrize(
    ("env_name", "num_bandits", "episode_length"),
    [
        ("popgym_multiarmed_bandit_medium", 20, 400),
        ("popgym_multiarmed_bandit_hard", 30, 600),
    ],
)
def test_multiarmed_bandit_variant_metadata_and_reward_scale(
    env_name, num_bandits, episode_length
):
    C = _skip_if_wrong_env(env_name)
    args = {"vec": {"total_agents": 1, "num_buffers": 1, "num_threads": 1}, "env": {}}
    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        assert vec.obs_size == 1
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [num_bandits]

        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 1), dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        assert abs(rewards[0]) == pytest.approx(1.0 / episode_length)
        assert terminals[0] == 0.0
    finally:
        vec.close()
