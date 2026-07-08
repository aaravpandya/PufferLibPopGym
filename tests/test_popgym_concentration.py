import ctypes

import numpy as np
import pytest


def _skip_if_wrong_env():
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != "popgym_concentration":
        pytest.skip(
            "Build the target env first: "
            "PYTHON=.venv/bin/python ./build.sh popgym_concentration --cpu"
        )
    return C


def _make_vec(total_agents=1):
    C = _skip_if_wrong_env()
    args = {
        "vec": {"total_agents": total_agents, "num_buffers": 1, "num_threads": 1},
        "env": {},
    }
    vec = C.create_vec(args, 0)
    vec.reset()
    return vec


def test_concentration_smoke():
    vec = _make_vec(total_agents=16)
    try:
        assert vec.obs_size == 52
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [52]

        obs = np.ctypeslib.as_array(
            (ctypes.c_ubyte * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
        ).reshape(vec.total_agents, vec.obs_size)
        rewards = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
        )
        terminals = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
        )
        actions = np.random.randint(0, 52, size=(vec.total_agents, 1)).astype(np.float32)

        assert np.all(obs == 13)
        for _ in range(20):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all(obs <= 13)
            assert rewards.shape == (vec.total_agents,)
            assert terminals.shape == (vec.total_agents,)
    finally:
        vec.close()


def test_concentration_same_card_twice_penalty_and_obs_timing():
    vec = _make_vec()
    try:
        obs = np.ctypeslib.as_array((ctypes.c_ubyte * vec.obs_size).from_address(vec.obs_ptr))
        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 1), dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == 0.0
        assert terminals[0] == 0.0
        first_rank = int(obs[0])
        assert first_rank < 13
        assert np.all(obs[1:] == 13)

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-2.0 / 104.0)
        assert terminals[0] == 0.0
        assert obs[0] == first_rank

        action[0, 0] = 1
        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == 0.0
        assert obs[0] == 13
        assert obs[1] < 13
    finally:
        vec.close()


def test_concentration_terminal_step_preserves_reward():
    vec = _make_vec()
    try:
        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 1), dtype=np.float32)

        for tick in range(104):
            vec.cpu_step(action.ctypes.data)
            if terminals[0]:
                assert tick == 103
                assert rewards[0] == pytest.approx(-2.0 / 104.0)
                break
        else:
            pytest.fail("episode did not truncate after 104 steps")
    finally:
        vec.close()
