import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


def test_count_recall_smoke():
    with vec_for("popgym_count_recall", total_agents=16) as vec:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [27]

        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        actions = np.random.randint(0, 27, size=(vec.total_agents, 1)).astype(np.float32)

        for _ in range(40):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all(obs < 2)
            assert np.allclose(np.abs(rewards), 1.0 / 51.0)
            assert np.all(terminals == 0.0)


def test_count_recall_rewards_previous_query_count():
    with vec_for("popgym_count_recall") as vec:
        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
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


def test_count_recall_wrong_count_penalty():
    with vec_for("popgym_count_recall") as vec:
        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)
        counts = [0, 0]
        counts[int(obs[0, 0])] += 1

        prev_query = int(obs[0, 1])
        action[0, 0] = counts[prev_query] + 1
        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-1.0 / 51.0)
        assert terminals[0] == 0.0


def test_count_recall_terminal_step_preserves_reward():
    with vec_for("popgym_count_recall") as vec:
        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
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
