import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for

ENV_KWARGS = {"num_decks": 1, "k": 4, "include_prev_action": 0, "include_antialias": 0}


def test_repeat_previous_smoke():
    with vec_for("popgym_repeat_previous", total_agents=16, env_kwargs=ENV_KWARGS) as vec:
        assert vec.obs_size == 4
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [4]

        obs = obs_array(vec)
        assert obs.shape == (vec.total_agents, vec.obs_size)
        assert obs.dtype == np.uint8

        actions = np.random.randint(0, 4, size=(vec.total_agents, vec.num_atns), dtype=np.int32).astype(
            np.float32
        )
        rewards = np.empty((vec.total_agents,), dtype=np.float32)
        terminals = np.empty((vec.total_agents,), dtype=np.float32)
        for _ in range(100):
            vec.cpu_step(actions.ctypes.data)
            rewards[:] = rewards_array(vec)
            terminals[:] = terminals_array(vec)
            assert rewards.shape == (vec.total_agents,)
            assert terminals.shape == (vec.total_agents,)


def test_repeat_previous_reward_starts_when_k_cards_are_visible():
    with vec_for("popgym_repeat_previous", env_kwargs=ENV_KWARGS) as vec:
        obs = obs_array(vec)[0]
        reward = rewards_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)
        seen = [int(obs[0])]

        for _ in range(3):
            vec.cpu_step(action.ctypes.data)
            assert reward[0] == 0.0
            seen.append(int(obs[0]))

        action[0, 0] = seen[0]
        vec.cpu_step(action.ctypes.data)
        assert reward[0] == pytest.approx(1.0 / 48.0)


def test_repeat_previous_terminal_step_preserves_reward():
    with vec_for("popgym_repeat_previous", env_kwargs=ENV_KWARGS) as vec:
        obs = obs_array(vec)[0]
        reward = rewards_array(vec)
        terminal = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)
        seen = [int(obs[0])]

        for tick in range(51):
            if tick + 1 >= 4:
                action[0, 0] = seen[tick + 1 - 4]
            vec.cpu_step(action.ctypes.data)
            if terminal[0]:
                assert tick == 50
                assert reward[0] == pytest.approx(1.0 / 48.0)
                break
            seen.append(int(obs[0]))
        else:
            pytest.fail("episode did not terminate after 51 steps")
