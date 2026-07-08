import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


def test_velocity_only_cartpole_smoke_and_reset_contract():
    with vec_for(
        "popgym_velocity_only_cartpole",
        total_agents=16,
        env_kwargs={"max_episode_length": 200},
    ) as vec:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [2]

        obs = obs_array(vec, dtype="float32")
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        actions = np.random.randint(0, 2, size=(vec.total_agents, 1)).astype(np.float32)

        # POPGym currently returns [x, theta] on reset for this env.
        assert np.all(np.abs(obs[:, 0]) <= 0.05)
        assert np.all(np.abs(obs[:, 1]) <= 0.05)
        vec.cpu_step(actions.ctypes.data)
        assert np.any(np.abs(obs) > 0.05)
        assert np.all(np.isfinite(obs))
        assert np.allclose(rewards, 1.0 / 200.0)
        assert rewards.shape == (vec.total_agents,)
        assert terminals.shape == (vec.total_agents,)


def test_velocity_only_cartpole_timeout_preserves_reward():
    with vec_for(
        "popgym_velocity_only_cartpole",
        env_kwargs={"max_episode_length": 3},
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
