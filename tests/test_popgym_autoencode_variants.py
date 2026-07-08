import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


@pytest.mark.parametrize(
    ("env_name", "num_decks"),
    [
        ("popgym_autoencode_easy", 1),
        ("popgym_autoencode_medium", 2),
        ("popgym_autoencode_hard", 3),
    ],
)
def test_autoencode_variant_phase_lengths_and_reward_scale(env_name, num_decks):
    num_cards = 52 * num_decks
    with vec_for(env_name) as vec:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [4]

        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        assert obs[0, 0] == 1  # watch phase at reset

        for _ in range(num_cards - 1):
            vec.cpu_step(action.ctypes.data)
            assert rewards[0] == 0.0
            assert terminals[0] == 0.0
        assert obs[0, 0] == 0  # play phase after watching the whole deck

        for step in range(1, num_cards + 1):
            vec.cpu_step(action.ctypes.data)
            assert abs(rewards[0]) == pytest.approx(1.0 / num_cards)
            expected_terminal = 1.0 if step == num_cards else 0.0
            assert terminals[0] == expected_terminal
