import numpy as np
import pytest

from tests.popgym_helpers import (
        obs_array,
        rewards_array,
        steps_until_terminal,
        terminals_array,
        vec_for,
    )


def test_concentration_smoke():
    with vec_for("popgym_concentration", total_agents=16) as vec:
        assert vec.obs_size == 52
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [52]

        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        actions = np.random.randint(0, 52, size=(vec.total_agents, 1)).astype(np.float32)

        assert np.all(obs == 13)
        for _ in range(20):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all(obs <= 13)
            assert rewards.shape == (vec.total_agents,)
            assert terminals.shape == (vec.total_agents,)


def test_concentration_same_card_twice_penalty_and_obs_timing():
    with vec_for("popgym_concentration") as vec:
        obs = obs_array(vec)[0]
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == 0.0
        assert terminals[0] == 0.0
        first_rank = int(obs[0])
        assert first_rank < 13
        assert np.all(obs[1:] == 13)

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-2.0 / 104.0)
        assert terminals[0] == 0.0
        assert obs[0] == first_rank

        action[0, 0] = 1
        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == 0.0
        assert obs[0] == 13
        assert obs[1] < 13


def test_concentration_terminal_step_preserves_reward():
    with vec_for("popgym_concentration") as vec:
        rewards = rewards_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        steps = steps_until_terminal(vec, action, 104)
        assert steps == 104
        assert rewards[0] == pytest.approx(-2.0 / 104.0)
