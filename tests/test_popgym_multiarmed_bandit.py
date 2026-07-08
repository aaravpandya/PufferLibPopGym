import ctypes

import numpy as np
import pytest


def _skip_if_wrong_env():
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != "popgym_multiarmed_bandit":
        pytest.skip(
            "Build the target env first: "
            "PYTHON=.venv/bin/python ./build.sh popgym_multiarmed_bandit --cpu"
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


def test_multiarmed_bandit_smoke():
    vec = _make_vec(total_agents=16)
    try:
        assert vec.obs_size == 1
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [10]

        obs = np.ctypeslib.as_array(
            (ctypes.c_ubyte * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
        ).reshape(vec.total_agents, vec.obs_size)
        rewards = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
        )
        terminals = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
        )
        actions = np.random.randint(0, 10, size=(vec.total_agents, 1)).astype(np.float32)

        for _ in range(40):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all((obs == 0) | (obs == 1))
            assert np.allclose(np.abs(rewards), 1.0 / 200.0)
            assert np.all(terminals == 0.0)
    finally:
        vec.close()


def test_multiarmed_bandit_terminal_step_preserves_reward():
    vec = _make_vec()
    try:
        obs = np.ctypeslib.as_array((ctypes.c_ubyte * vec.obs_size).from_address(vec.obs_ptr))
        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 1), dtype=np.float32)

        assert obs[0] == 0
        for tick in range(200):
            vec.cpu_step(action.ctypes.data)
            assert abs(rewards[0]) == pytest.approx(1.0 / 200.0)
            if terminals[0]:
                assert tick == 199
                break
        else:
            pytest.fail("episode did not terminate after 200 steps")
    finally:
        vec.close()
