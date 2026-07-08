import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


def test_position_only_pendulum_smoke():
    with vec_for(
        "popgym_position_only_pendulum",
        total_agents=16,
        env_kwargs={"max_episode_length": 200},
    ) as vec:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [1]

        obs = obs_array(vec, dtype="float32")
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        actions = np.random.uniform(-2, 2, size=(vec.total_agents, 1)).astype(np.float32)

        assert np.all(np.isclose(np.linalg.norm(obs, axis=1), 1.0, atol=1e-5))
        for _ in range(20):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.float32
            assert np.all(np.isfinite(obs))
            assert np.all(np.abs(obs) <= 1.0)
            assert np.all(rewards <= 1.0 / 200.0 + 1e-6)
            assert np.all(rewards >= -1.0 / 200.0 - 1e-6)
            assert terminals.shape == (vec.total_agents,)


def test_position_only_pendulum_timeout_preserves_reward():
    with vec_for(
        "popgym_position_only_pendulum",
        env_kwargs={"max_episode_length": 3},
    ) as vec:
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        for tick in range(3):
            vec.cpu_step(action.ctypes.data)
            assert -1.0 / 3.0 <= rewards[0] <= 1.0 / 3.0
            if terminals[0]:
                assert tick == 2
                break
        else:
            pytest.fail("episode did not truncate after max_episode_length")
