import numpy as np
import pytest

from tests.popgym_helpers import rewards_array, terminals_array, vec_for


@pytest.mark.parametrize(
    ("env_name", "num_decks"),
    [
        ("popgym_higher_lower_easy", 1),
        ("popgym_higher_lower_medium", 2),
        ("popgym_higher_lower_hard", 3),
    ],
)
def test_higher_lower_variant_episode_length_and_reward_scale(env_name, num_decks):
    num_cards = 52 * num_decks
    with vec_for(env_name) as vec:
        assert vec.obs_size == 1
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [2]

        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        for step in range(1, num_cards):
            vec.cpu_step(action.ctypes.data)
            assert np.isclose(rewards[0], 0.0) or np.isclose(
                abs(rewards[0]), 1.0 / num_cards
            )
            expected_terminal = 1.0 if step == num_cards - 1 else 0.0
            assert terminals[0] == expected_terminal
