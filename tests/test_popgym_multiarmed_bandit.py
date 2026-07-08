import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


def test_multiarmed_bandit_smoke():
    with vec_for("popgym_multiarmed_bandit", total_agents=16) as vec:
        assert vec.obs_size == 1
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [10]

        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        actions = np.random.randint(0, 10, size=(vec.total_agents, 1)).astype(np.float32)

        for _ in range(40):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all((obs == 0) | (obs == 1))
            assert np.allclose(np.abs(rewards), 1.0 / 200.0)
            assert np.all(terminals == 0.0)


def test_multiarmed_bandit_terminal_step_preserves_reward():
    with vec_for("popgym_multiarmed_bandit") as vec:
        obs = obs_array(vec)[0]
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        assert obs[0] == 0
        for tick in range(200):
            vec.cpu_step(action.ctypes.data)
            assert abs(rewards[0]) == pytest.approx(1.0 / 200.0)
            if terminals[0]:
                assert tick == 199
                break
        else:
            pytest.fail("episode did not terminate after 200 steps")
