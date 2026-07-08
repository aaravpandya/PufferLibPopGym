import ctypes

import numpy as np
import pytest


def _skip_if_wrong_env():
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != "popgym_battleship":
        pytest.skip(
            "Build the target env first: "
            "PYTHON=.venv/bin/python ./build.sh popgym_battleship --cpu"
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


def test_battleship_smoke():
    vec = _make_vec(total_agents=16)
    try:
        assert vec.obs_size == 1
        assert vec.num_atns == 2
        assert list(vec.act_sizes) == [10, 10]

        obs = np.ctypeslib.as_array(
            (ctypes.c_ubyte * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
        ).reshape(vec.total_agents, vec.obs_size)
        rewards = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
        )
        terminals = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
        )
        actions = np.random.randint(0, 10, size=(vec.total_agents, 2)).astype(np.float32)

        for _ in range(20):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all((obs == 0) | (obs == 1))
            assert np.all(
                np.isclose(rewards, 1.0 / 12.0)
                | np.isclose(rewards, -1.0 / 88.0)
            )
            assert rewards.shape == (vec.total_agents,)
            assert terminals.shape == (vec.total_agents,)
    finally:
        vec.close()


def test_battleship_repeat_guess_is_miss():
    vec = _make_vec()
    try:
        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 2), dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        first_reward = rewards[0]
        assert np.isclose(first_reward, 1.0 / 12.0) or np.isclose(first_reward, -1.0 / 88.0)
        assert terminals[0] == 0.0

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-1.0 / 88.0)
        assert terminals[0] == 0.0
    finally:
        vec.close()


def test_battleship_exhaustive_scan_terminates_by_board_limit():
    vec = _make_vec()
    try:
        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 2), dtype=np.float32)

        for row in range(10):
            for col in range(10):
                action[0, 0] = row
                action[0, 1] = col
                vec.cpu_step(action.ctypes.data)
                assert np.isclose(rewards[0], 1.0 / 12.0) or np.isclose(rewards[0], -1.0 / 88.0)
                if terminals[0]:
                    return
        pytest.fail("episode did not terminate during exhaustive scan")
    finally:
        vec.close()
