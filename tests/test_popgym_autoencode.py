import ctypes

import numpy as np
import pytest


def _skip_if_wrong_env():
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != "popgym_autoencode":
        pytest.skip(
            "Build the target env first: "
            "PYTHON=.venv/bin/python ./build.sh popgym_autoencode --cpu"
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


def test_autoencode_smoke():
    vec = _make_vec(total_agents=16)
    try:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [4]

        obs, rewards, terminals = _arrays(vec)
        actions = np.random.randint(0, 4, size=(vec.total_agents, 1)).astype(np.float32)

        for _ in range(120):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all((obs[:, 0] == 0) | (obs[:, 0] == 1))
            assert np.all(obs[:, 1] < 4)
            assert rewards.shape == (vec.total_agents,)
            assert terminals.shape == (vec.total_agents,)
    finally:
        vec.close()


def test_autoencode_watch_then_recite_reverse():
    vec = _make_vec()
    try:
        obs, rewards, terminals = _arrays(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        seen = [int(obs[0, 1])]
        assert obs[0, 0] == 1

        for watch_step in range(51):
            vec.cpu_step(action.ctypes.data)
            assert rewards[0] == 0.0
            assert terminals[0] == 0.0
            seen.append(int(obs[0, 1]))

            expected_mode = 0 if watch_step == 50 else 1
            assert obs[0, 0] == expected_mode

        assert len(seen) == 52
        assert obs[0, 0] == 0

        for play_step, target in enumerate(reversed(seen)):
            action[0, 0] = target
            vec.cpu_step(action.ctypes.data)
            assert rewards[0] == pytest.approx(1.0 / 52.0)
            if terminals[0]:
                assert play_step == 51
                break
            assert obs[0, 0] == 0
            assert obs[0, 1] == 0
        else:
            pytest.fail("episode did not terminate after reciting all cards")
    finally:
        vec.close()


def test_autoencode_wrong_action_penalty():
    vec = _make_vec()
    try:
        obs, rewards, terminals = _arrays(vec)
        action = np.zeros((1, 1), dtype=np.float32)
        seen = [int(obs[0, 1])]

        for _ in range(51):
            vec.cpu_step(action.ctypes.data)
            seen.append(int(obs[0, 1]))

        action[0, 0] = (seen[-1] + 1) % 4
        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-1.0 / 52.0)
        assert terminals[0] == 0.0
    finally:
        vec.close()


def test_autoencode_num_decks_controls_episode_length():
    vec = _make_vec(num_decks=2)
    try:
        obs, rewards, terminals = _arrays(vec)
        action = np.zeros((1, 1), dtype=np.float32)
        seen = [int(obs[0, 1])]

        for _ in range(103):
            vec.cpu_step(action.ctypes.data)
            seen.append(int(obs[0, 1]))

        assert len(seen) == 104
        assert obs[0, 0] == 0

        for play_step, target in enumerate(reversed(seen)):
            action[0, 0] = target
            vec.cpu_step(action.ctypes.data)
            if terminals[0]:
                assert play_step == 103
                assert rewards[0] == pytest.approx(1.0 / 104.0)
                break
        else:
            pytest.fail("episode did not terminate after 207 total steps")
    finally:
        vec.close()
