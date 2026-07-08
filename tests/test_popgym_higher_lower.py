import ctypes

import numpy as np
import pytest


def _skip_if_wrong_env():
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != "popgym_higher_lower":
        pytest.skip(
            "Build the target env first: "
            "PYTHON=.venv/bin/python ./build.sh popgym_higher_lower --cpu"
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


def test_higher_lower_smoke():
    vec = _make_vec(total_agents=16)
    try:
        assert vec.obs_size == 1
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [2]

        obs = np.ctypeslib.as_array(
            (ctypes.c_ubyte * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
        ).reshape(vec.total_agents, vec.obs_size)
        rewards = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
        )
        terminals = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
        )
        actions = np.random.randint(0, 2, size=(vec.total_agents, 1)).astype(np.float32)

        for _ in range(40):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all(obs < 13)
            assert np.all((np.isclose(rewards, 0.0)) | np.isclose(np.abs(rewards), 1.0 / 52.0))
            assert np.all(terminals == 0.0)
    finally:
        vec.close()


def test_higher_lower_reward_matches_observed_rank_change():
    vec = _make_vec()
    try:
        obs = np.ctypeslib.as_array((ctypes.c_ubyte * vec.obs_size).from_address(vec.obs_ptr))
        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 1), dtype=np.float32)

        for _ in range(30):
            prev_rank = int(obs[0])
            action[0, 0] = 0
            vec.cpu_step(action.ctypes.data)
            next_rank = int(obs[0])
            if next_rank == prev_rank:
                assert rewards[0] == 0.0
            elif next_rank > prev_rank:
                assert rewards[0] == pytest.approx(1.0 / 52.0)
            else:
                assert rewards[0] == pytest.approx(-1.0 / 52.0)
            assert terminals[0] == 0.0
    finally:
        vec.close()


def test_higher_lower_terminal_step_preserves_reward():
    vec = _make_vec()
    try:
        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 1), dtype=np.float32)

        for tick in range(51):
            vec.cpu_step(action.ctypes.data)
            assert np.isclose(rewards[0], 0.0) or abs(rewards[0]) == pytest.approx(1.0 / 52.0)
            if terminals[0]:
                assert tick == 50
                break
        else:
            pytest.fail("episode did not terminate after 51 steps")
    finally:
        vec.close()


def test_higher_lower_num_decks_controls_episode_length_and_reward():
    vec = _make_vec(num_decks=2)
    try:
        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
        action = np.zeros((1, 1), dtype=np.float32)

        for tick in range(103):
            vec.cpu_step(action.ctypes.data)
            assert np.isclose(rewards[0], 0.0) or abs(rewards[0]) == pytest.approx(1.0 / 104.0)
            if terminals[0]:
                assert tick == 102
                break
        else:
            pytest.fail("episode did not terminate after 103 steps")
    finally:
        vec.close()
