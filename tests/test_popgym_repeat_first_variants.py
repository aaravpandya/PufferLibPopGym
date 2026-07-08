import numpy as np
import pytest

from tests.popgym_helpers import rewards_array, terminals_array, vec_for


@pytest.mark.parametrize(
    ("env_name", "num_decks"),
    [
        ("popgym_repeat_first_easy", 1),
        ("popgym_repeat_first_medium", 8),
        ("popgym_repeat_first_hard", 16),
    ],
)
def test_repeat_first_variant_episode_length_and_reward_scale(env_name, num_decks):
    episode_length = 52 * num_decks - 1
    with vec_for(env_name) as vec:
        assert vec.obs_size == 1
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [4]

        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        for step in range(1, episode_length + 1):
            vec.cpu_step(action.ctypes.data)
            assert abs(rewards[0]) == pytest.approx(1.0 / episode_length)
            expected_terminal = 1.0 if step == episode_length else 0.0
            assert terminals[0] == expected_terminal
