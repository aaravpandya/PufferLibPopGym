import numpy as np
import pytest

from tests.popgym_helpers import rewards_array, terminals_array, vec_for


@pytest.mark.parametrize(
    ("env_name", "rows", "cols", "mines"),
    [
        ("popgym_minesweeper_easy", 4, 4, 2),
        ("popgym_minesweeper_medium", 6, 6, 6),
        ("popgym_minesweeper_hard", 8, 8, 10),
    ],
)
def test_minesweeper_variant_metadata_and_invalid_action_reward(env_name, rows, cols, mines):
    with vec_for(env_name) as vec:
        max_steps = rows * cols - mines
        assert vec.obs_size == 1
        assert vec.num_atns == 2
        assert list(vec.act_sizes) == [rows, cols]

        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.array([[rows, 0]], dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-0.5 / (max_steps - 2))
        assert terminals[0] == 0.0
