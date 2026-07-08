import ctypes

import numpy as np
import pytest


def _skip_if_wrong_env():
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != "popgym_count_recall":
        pytest.skip(
            "Build the target env first: "
            "PYTHON=.venv/bin/python ./build.sh popgym_count_recall --cpu"
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


def _arrays(vec):
    obs = np.ctypeslib.as_array(
        (ctypes.c_ubyte * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
    ).reshape(vec.total_agents, vec.obs_size)
    rewards = np.ctypeslib.as_array(
        (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
    )
    terminals = np.ctypeslib.as_array(
        (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
    )
    return obs, rewards, terminals


def test_count_recall_smoke():
    vec = _make_vec(total_agents=16)
    try:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [27]

        obs, rewards, terminals = _arrays(vec)
        actions = np.random.randint(0, 27, size=(vec.total_agents, 1)).astype(np.float32)

        for _ in range(40):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all(obs < 2)
            assert np.allclose(np.abs(rewards), 1.0 / 51.0)
            assert np.all(terminals == 0.0)
    finally:
        vec.close()


def test_count_recall_rewards_previous_query_count():
    vec = _make_vec()
    try:
        obs, rewards, terminals = _arrays(vec)
        action = np.zeros((1, 1), dtype=np.float32)
        counts = [0, 0]
        counts[int(obs[0, 0])] += 1

        for _ in range(30):
            prev_query = int(obs[0, 1])
            action[0, 0] = counts[prev_query]
            vec.cpu_step(action.ctypes.data)
            assert rewards[0] == pytest.approx(1.0 / 51.0)
            assert terminals[0] == 0.0
            counts[int(obs[0, 0])] += 1
    finally:
        vec.close()


def test_count_recall_wrong_count_penalty():
    vec = _make_vec()
    try:
        obs, rewards, terminals = _arrays(vec)
        action = np.zeros((1, 1), dtype=np.float32)
        counts = [0, 0]
        counts[int(obs[0, 0])] += 1

        prev_query = int(obs[0, 1])
        action[0, 0] = counts[prev_query] + 1
        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-1.0 / 51.0)
        assert terminals[0] == 0.0
    finally:
        vec.close()


def test_count_recall_terminal_step_preserves_reward():
    vec = _make_vec()
    try:
        obs, rewards, terminals = _arrays(vec)
        action = np.zeros((1, 1), dtype=np.float32)
        counts = [0, 0]
        counts[int(obs[0, 0])] += 1

        for tick in range(51):
            prev_query = int(obs[0, 1])
            action[0, 0] = counts[prev_query]
            vec.cpu_step(action.ctypes.data)
            assert rewards[0] == pytest.approx(1.0 / 51.0)
            if terminals[0]:
                assert tick == 50
                break
            counts[int(obs[0, 0])] += 1
        else:
            pytest.fail("episode did not terminate after 51 steps")
    finally:
        vec.close()
