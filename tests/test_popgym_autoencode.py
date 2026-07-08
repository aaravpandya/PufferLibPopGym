import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


def test_autoencode_smoke():
    with vec_for("popgym_autoencode", total_agents=16) as vec:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [4]

        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        actions = np.random.randint(0, 4, size=(vec.total_agents, 1)).astype(np.float32)

        for _ in range(120):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all((obs[:, 0] == 0) | (obs[:, 0] == 1))
            assert np.all(obs[:, 1] < 4)
            assert rewards.shape == (vec.total_agents,)
            assert terminals.shape == (vec.total_agents,)


def test_autoencode_watch_then_recite_reverse():
    with vec_for("popgym_autoencode") as vec:
        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
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


def test_autoencode_wrong_action_penalty():
    with vec_for("popgym_autoencode") as vec:
        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)
        seen = [int(obs[0, 1])]

        for _ in range(51):
            vec.cpu_step(action.ctypes.data)
            seen.append(int(obs[0, 1]))

        action[0, 0] = (seen[-1] + 1) % 4
        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-1.0 / 52.0)
        assert terminals[0] == 0.0


def test_autoencode_num_decks_controls_episode_length():
    with vec_for("popgym_autoencode", env_kwargs={"num_decks": 2}) as vec:
        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
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
