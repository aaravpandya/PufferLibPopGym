import ctypes
import numpy as np
import pytest


def _skip_if_wrong_env():
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != "popgym_repeat_previous":
        pytest.skip(
            "Build the target env first: "
            "PYTHON=.venv/bin/python ./build.sh popgym_repeat_previous --cpu"
        )
    return C


def test_repeat_previous_smoke():
    C = _skip_if_wrong_env()

    args = {
        "vec": {"total_agents": 16, "num_buffers": 1, "num_threads": 1},
        "env": {"num_decks": 1, "k": 4, "include_prev_action": 0, "include_antialias": 0},
    }

    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        assert vec.obs_size == 4
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [4]

        obs_ptr = (ctypes.c_ubyte * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
        obs = np.ctypeslib.as_array(obs_ptr).reshape(vec.total_agents, vec.obs_size)
        assert obs.shape == (vec.total_agents, vec.obs_size)
        assert obs.dtype == np.uint8

        actions = np.random.randint(0, 4, size=(vec.total_agents, vec.num_atns), dtype=np.int32).astype(
            np.float32
        )
        rewards = np.empty((vec.total_agents,), dtype=np.float32)
        terminals = np.empty((vec.total_agents,), dtype=np.float32)
        for _ in range(100):
            vec.cpu_step(actions.ctypes.data)
            rewards[:] = np.ctypeslib.as_array(
                (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
            )
            terminals[:] = np.ctypeslib.as_array(
                (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
            )
            assert rewards.shape == (vec.total_agents,)
            assert terminals.shape == (vec.total_agents,)
    finally:
        vec.close()


def test_repeat_previous_reward_starts_when_k_cards_are_visible():
    C = _skip_if_wrong_env()

    args = {
        "vec": {"total_agents": 1, "num_buffers": 1, "num_threads": 1},
        "env": {"num_decks": 1, "k": 4, "include_prev_action": 0, "include_antialias": 0},
    }

    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        obs = np.ctypeslib.as_array(
            (ctypes.c_ubyte * vec.obs_size).from_address(vec.obs_ptr)
        )
        reward = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        action = np.zeros((1, 1), dtype=np.float32)
        seen = [int(obs[0])]

        for _ in range(3):
            vec.cpu_step(action.ctypes.data)
            assert reward[0] == 0.0
            seen.append(int(obs[0]))

        action[0, 0] = seen[0]
        vec.cpu_step(action.ctypes.data)
        assert reward[0] == pytest.approx(1.0 / 48.0)
    finally:
        vec.close()


def test_repeat_previous_terminal_step_preserves_reward():
    C = _skip_if_wrong_env()

    args = {
        "vec": {"total_agents": 1, "num_buffers": 1, "num_threads": 1},
        "env": {"num_decks": 1, "k": 4, "include_prev_action": 0, "include_antialias": 0},
    }

    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        obs = np.ctypeslib.as_array(
            (ctypes.c_ubyte * vec.obs_size).from_address(vec.obs_ptr)
        )
        reward = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminal = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 1), dtype=np.float32)
        seen = [int(obs[0])]

        for tick in range(51):
            if tick + 1 >= 4:
                action[0, 0] = seen[tick + 1 - 4]
            vec.cpu_step(action.ctypes.data)
            if terminal[0]:
                assert tick == 50
                assert reward[0] == pytest.approx(1.0 / 48.0)
                break
            seen.append(int(obs[0]))
        else:
            pytest.fail("episode did not terminate after 51 steps")
    finally:
        vec.close()
