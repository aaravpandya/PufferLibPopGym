import ctypes

import numpy as np
import pytest


def _skip_if_wrong_env():
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != "popgym_repeat_first":
        pytest.skip(
            "Build the target env first: "
            "PYTHON=.venv/bin/python ./build.sh popgym_repeat_first --cpu"
        )
    return C


def _make_vec(total_agents=1, num_decks=1):
    C = _skip_if_wrong_env()
    args = {
        "vec": {"total_agents": total_agents, "num_buffers": 1, "num_threads": 1},
        "env": {"num_decks": num_decks},
    }
    vec = C.create_vec(args, 0)
    vec.reset()
    return vec


def test_repeat_first_smoke():
    vec = _make_vec(total_agents=16)
    try:
        assert vec.obs_size == 1
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [4]

        obs = np.ctypeslib.as_array(
            (ctypes.c_ubyte * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
        ).reshape(vec.total_agents, vec.obs_size)
        rewards = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
        )
        terminals = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
        )
        actions = np.random.randint(0, 4, size=(vec.total_agents, 1)).astype(np.float32)

        for _ in range(100):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all(obs >= 0)
            assert np.all(obs < 4)
            assert rewards.shape == (vec.total_agents,)
            assert terminals.shape == (vec.total_agents,)
    finally:
        vec.close()


def test_repeat_first_rewards_repeat_initial_suit():
    vec = _make_vec()
    try:
        obs = np.ctypeslib.as_array((ctypes.c_ubyte * vec.obs_size).from_address(vec.obs_ptr))
        reward = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminal = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 1), dtype=np.float32)

        first_suit = int(obs[0])
        action[0, 0] = first_suit
        vec.cpu_step(action.ctypes.data)
        assert reward[0] == pytest.approx(1.0 / 51.0)
        assert terminal[0] == 0.0

        wrong_suit = (first_suit + 1) % 4
        action[0, 0] = wrong_suit
        vec.cpu_step(action.ctypes.data)
        assert reward[0] == pytest.approx(-1.0 / 51.0)
        assert terminal[0] == 0.0
    finally:
        vec.close()


def test_repeat_first_terminal_step_preserves_reward():
    vec = _make_vec()
    try:
        obs = np.ctypeslib.as_array((ctypes.c_ubyte * vec.obs_size).from_address(vec.obs_ptr))
        reward = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminal = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.array([[int(obs[0])]], dtype=np.float32)

        for tick in range(51):
            vec.cpu_step(action.ctypes.data)
            if terminal[0]:
                assert tick == 50
                assert reward[0] == pytest.approx(1.0 / 51.0)
                break
        else:
            pytest.fail("episode did not terminate after 51 steps")
    finally:
        vec.close()


def test_repeat_first_num_decks_controls_episode_length():
    vec = _make_vec(num_decks=2)
    try:
        obs = np.ctypeslib.as_array((ctypes.c_ubyte * vec.obs_size).from_address(vec.obs_ptr))
        reward = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminal = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.array([[int(obs[0])]], dtype=np.float32)

        for tick in range(103):
            vec.cpu_step(action.ctypes.data)
            if terminal[0]:
                assert tick == 102
                assert reward[0] == pytest.approx(1.0 / 103.0)
                break
        else:
            pytest.fail("episode did not terminate after 103 steps")
    finally:
        vec.close()
