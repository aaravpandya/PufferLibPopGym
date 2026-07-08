import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


def test_higher_lower_smoke():
    with vec_for(
        "popgym_higher_lower", total_agents=16, env_kwargs={"num_decks": 1}
    ) as vec:
        assert vec.obs_size == 1
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [2]

        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        actions = np.random.randint(0, 2, size=(vec.total_agents, 1)).astype(np.float32)

        for _ in range(40):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all(obs < 13)
            assert np.all((np.isclose(rewards, 0.0)) | np.isclose(np.abs(rewards), 1.0 / 52.0))
            assert np.all(terminals == 0.0)


def test_higher_lower_reward_matches_observed_rank_change():
    with vec_for("popgym_higher_lower", env_kwargs={"num_decks": 1}) as vec:
        obs = obs_array(vec)[0]
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
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


def test_higher_lower_terminal_step_preserves_reward():
    with vec_for("popgym_higher_lower", env_kwargs={"num_decks": 1}) as vec:
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        for tick in range(51):
            vec.cpu_step(action.ctypes.data)
            assert np.isclose(rewards[0], 0.0) or abs(rewards[0]) == pytest.approx(1.0 / 52.0)
            if terminals[0]:
                assert tick == 50
                break
        else:
            pytest.fail("episode did not terminate after 51 steps")


def test_higher_lower_num_decks_controls_episode_length_and_reward():
    with vec_for("popgym_higher_lower", env_kwargs={"num_decks": 2}) as vec:
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        for tick in range(103):
            vec.cpu_step(action.ctypes.data)
            assert np.isclose(rewards[0], 0.0) or abs(rewards[0]) == pytest.approx(1.0 / 104.0)
            if terminals[0]:
                assert tick == 102
                break
        else:
            pytest.fail("episode did not terminate after 103 steps")
