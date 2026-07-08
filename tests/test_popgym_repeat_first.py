import numpy as np
import pytest

from tests.popgym_helpers import (
        obs_array,
        rewards_array,
        steps_until_terminal,
        terminals_array,
        vec_for,
    )


def test_repeat_first_smoke():
    with vec_for("popgym_repeat_first", total_agents=16) as vec:
        assert vec.obs_size == 1
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [4]

        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        actions = np.random.randint(0, 4, size=(vec.total_agents, 1)).astype(np.float32)

        for _ in range(100):
            vec.cpu_step(actions.ctypes.data)
            assert obs.shape == (vec.total_agents, vec.obs_size)
            assert obs.dtype == np.uint8
            assert np.all(obs >= 0)
            assert np.all(obs < 4)
            assert rewards.shape == (vec.total_agents,)
            assert terminals.shape == (vec.total_agents,)


def test_repeat_first_rewards_repeat_initial_suit():
    with vec_for("popgym_repeat_first") as vec:
        obs = obs_array(vec)[0]
        reward = rewards_array(vec)
        terminal = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        first_suit = int(obs[0])
        action[0, 0] = first_suit
        vec.cpu_step(action.ctypes.data)
        assert reward[0] == pytest.approx(1.0 / 51.0)
        assert terminal[0] == 0.0

        wrong_suit = (first_suit + 1) % 4
        action[0, 0] = wrong_suit
        vec.cpu_step(action.ctypes.data)
        assert reward[0] == pytest.approx(-1.0 / 51.0)
        assert terminal[0] == 0.0


def test_repeat_first_terminal_step_preserves_reward():
    with vec_for("popgym_repeat_first") as vec:
        obs = obs_array(vec)[0]
        reward = rewards_array(vec)
        action = np.array([[int(obs[0])]], dtype=np.float32)

        steps = steps_until_terminal(vec, action, 51)
        assert steps == 51
        assert reward[0] == pytest.approx(1.0 / 51.0)


def test_repeat_first_num_decks_controls_episode_length():
    with vec_for("popgym_repeat_first", env_kwargs={"num_decks": 2}) as vec:
        obs = obs_array(vec)[0]
        reward = rewards_array(vec)
        action = np.array([[int(obs[0])]], dtype=np.float32)

        steps = steps_until_terminal(vec, action, 103)
        assert steps == 103
        assert reward[0] == pytest.approx(1.0 / 103.0)
