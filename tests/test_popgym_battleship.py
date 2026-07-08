import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


def test_battleship_smoke():
    with vec_for("popgym_battleship", total_agents=16) as vec:
        assert vec.obs_size == 1
        assert vec.num_atns == 2
        assert list(vec.act_sizes) == [10, 10]

        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
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


def test_battleship_repeat_guess_is_miss():
    with vec_for("popgym_battleship") as vec:
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 2), dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        first_reward = rewards[0]
        assert np.isclose(first_reward, 1.0 / 12.0) or np.isclose(first_reward, -1.0 / 88.0)
        assert terminals[0] == 0.0

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-1.0 / 88.0)
        assert terminals[0] == 0.0


def test_battleship_exhaustive_scan_terminates_by_board_limit():
    with vec_for("popgym_battleship") as vec:
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
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
