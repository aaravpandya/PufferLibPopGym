import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for

ENV_NAME = "popgym_noisy_position_only_cartpole"


def test_noisy_position_only_cartpole_smoke():
    with vec_for(
        ENV_NAME,
        total_agents=16,
        env_kwargs={"max_episode_length": 200, "noise_sigma": 0.1},
    ) as vec:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [2]

        obs = obs_array(vec, dtype="float32")
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        actions = np.random.randint(0, 2, size=(vec.total_agents, 1)).astype(np.float32)

        assert np.all(obs[:, 0] >= -4.8)
        assert np.all(obs[:, 0] <= 4.8)
        assert np.all(obs[:, 1] >= -0.41887903)
        assert np.all(obs[:, 1] <= 0.41887903)
        for _ in range(20):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs))
            assert np.all(obs[:, 0] >= -4.8)
            assert np.all(obs[:, 0] <= 4.8)
            assert np.all(obs[:, 1] >= -0.41887903)
            assert np.all(obs[:, 1] <= 0.41887903)
            assert np.allclose(rewards, 1.0 / 200.0)
            assert terminals.shape == (vec.total_agents,)


def test_noisy_position_only_cartpole_zero_noise_matches_position_bounds():
    with vec_for(
        ENV_NAME,
        env_kwargs={"max_episode_length": 200, "noise_sigma": 0.0},
    ) as vec:
        obs = obs_array(vec, dtype="float32")[0]
        assert abs(obs[0]) <= 0.05
        assert abs(obs[1]) <= 0.05


def test_noisy_position_only_cartpole_timeout_preserves_reward():
    with vec_for(
        ENV_NAME,
        env_kwargs={"max_episode_length": 3, "noise_sigma": 0.1},
    ) as vec:
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        for tick in range(3):
            vec.cpu_step(action.ctypes.data)
            assert rewards[0] == pytest.approx(1.0 / 3.0)
            if terminals[0]:
                assert tick == 2
                break
        else:
            pytest.fail("episode did not truncate after max_episode_length")
