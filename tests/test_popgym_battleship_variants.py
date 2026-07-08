import numpy as np
import pytest

from tests.popgym_helpers import rewards_array, terminals_array, vec_for


@pytest.mark.parametrize(
    ("env_name", "board_size", "miss_reward"),
    [
        ("popgym_battleship_easy", 8, -1.0 / 52.0),
        ("popgym_battleship_medium", 10, -1.0 / 88.0),
        ("popgym_battleship_hard", 12, -1.0 / 132.0),
    ],
)
def test_battleship_variant_metadata_and_repeat_guess(env_name, board_size, miss_reward):
    with vec_for(env_name) as vec:
        assert vec.obs_size == 1
        assert vec.num_atns == 2
        assert list(vec.act_sizes) == [board_size, board_size]

        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 2), dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        assert np.isclose(rewards[0], 1.0 / 12.0) or np.isclose(rewards[0], miss_reward)
        assert terminals[0] == 0.0

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(miss_reward)
        assert terminals[0] == 0.0
