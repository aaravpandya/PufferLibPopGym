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
    ("env_name", "rows", "cols", "mines"),
    [
        ("popgym_minesweeper_medium", 6, 6, 6),
        ("popgym_minesweeper_hard", 8, 8, 10),
    ],
)
def test_minesweeper_variant_metadata_and_invalid_action_reward(env_name, rows, cols, mines):
    C = _skip_if_wrong_env(env_name)
    args = {"vec": {"total_agents": 1, "num_buffers": 1, "num_threads": 1}, "env": {}}
    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        max_steps = rows * cols - mines
        assert vec.obs_size == 1
        assert vec.num_atns == 2
        assert list(vec.act_sizes) == [rows, cols]

        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.array([[rows, 0]], dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-0.5 / (max_steps - 2))
        assert terminals[0] == 0.0
    finally:
        vec.close()
