import ctypes

import numpy as np
import pytest

from pufferlib.native_envs import load_native_env


def _load_or_skip(env_name):
    try:
        return load_native_env(env_name)
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))


def test_load_repeat_previous_and_minesweeper_side_by_side():
    repeat_previous = _load_or_skip("popgym_repeat_previous")
    minesweeper = _load_or_skip("popgym_minesweeper")

    assert repeat_previous.env_name == "popgym_repeat_previous"
    assert minesweeper.env_name == "popgym_minesweeper"
    assert repeat_previous is not minesweeper


def test_native_minesweeper_smoke():
    C = _load_or_skip("popgym_minesweeper")
    args = {
        "vec": {"total_agents": 16, "num_buffers": 1, "num_threads": 1},
        "env": {"difficulty": 0},
    }

    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        assert vec.obs_size == 1
        assert vec.num_atns == 2
        assert list(vec.act_sizes) == [4, 4]

        obs = np.ctypeslib.as_array(
            (ctypes.c_ubyte * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
        ).reshape(vec.total_agents, vec.obs_size)
        rewards = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
        )
        terminals = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
        )
        actions = np.zeros((vec.total_agents, vec.num_atns), dtype=np.float32)

        vec.cpu_step(actions.ctypes.data)
        assert obs.shape == (vec.total_agents, vec.obs_size)
        assert obs.dtype == np.uint8
        assert np.all(obs >= 0)
        assert np.all(obs <= 2)
        assert rewards.shape == (vec.total_agents,)
        assert terminals.shape == (vec.total_agents,)
    finally:
        vec.close()
