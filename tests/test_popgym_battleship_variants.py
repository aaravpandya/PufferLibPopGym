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
    ("env_name", "board_size", "miss_reward"),
    [
        ("popgym_battleship_easy", 8, -1.0 / 52.0),
        ("popgym_battleship_hard", 12, -1.0 / 132.0),
    ],
)
def test_battleship_variant_metadata_and_repeat_guess(env_name, board_size, miss_reward):
    C = _skip_if_wrong_env(env_name)
    args = {"vec": {"total_agents": 1, "num_buffers": 1, "num_threads": 1}, "env": {}}
    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        assert vec.obs_size == 1
        assert vec.num_atns == 2
        assert list(vec.act_sizes) == [board_size, board_size]

        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 2), dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        assert np.isclose(rewards[0], 1.0 / 12.0) or np.isclose(rewards[0], miss_reward)
        assert terminals[0] == 0.0

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(miss_reward)
        assert terminals[0] == 0.0
    finally:
        vec.close()
