import ctypes

import numpy as np
import pytest


def _skip_if_wrong_env():
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != "popgym_position_only_cartpole":
        pytest.skip(
            "Build the target env first: "
            "PYTHON=.venv/bin/python ./build.sh popgym_position_only_cartpole --cpu"
        )
    return C


def _make_vec(total_agents=1, max_episode_length=200):
    C = _skip_if_wrong_env()
    args = {
        "vec": {"total_agents": total_agents, "num_buffers": 1, "num_threads": 1},
        "env": {"max_episode_length": max_episode_length},
    }
    vec = C.create_vec(args, 0)
    vec.reset()
    return vec


def test_position_only_cartpole_smoke():
    vec = _make_vec(total_agents=16)
    try:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [2]

        obs = np.ctypeslib.as_array(
            (ctypes.c_float * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
        ).reshape(vec.total_agents, vec.obs_size)
        rewards = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
        )
        terminals = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
        )
        actions = np.random.randint(0, 2, size=(vec.total_agents, 1)).astype(np.float32)

        assert np.all(np.abs(obs[:, 0]) <= 0.05)
        assert np.all(np.abs(obs[:, 1]) <= 0.05)
        for _ in range(20):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs))
            assert np.allclose(rewards, 1.0 / 200.0)
            assert rewards.shape == (vec.total_agents,)
            assert terminals.shape == (vec.total_agents,)
    finally:
        vec.close()


def test_position_only_cartpole_timeout_preserves_reward():
    vec = _make_vec(max_episode_length=3)
    try:
        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 1), dtype=np.float32)

        for tick in range(3):
            vec.cpu_step(action.ctypes.data)
            assert rewards[0] == pytest.approx(1.0 / 3.0)
            if terminals[0]:
                assert tick == 2
                break
        else:
            pytest.fail("episode did not truncate after max_episode_length")
    finally:
        vec.close()
