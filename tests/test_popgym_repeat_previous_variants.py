import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


@pytest.mark.parametrize(
    ("env_name", "num_decks", "k"),
    [
        ("popgym_repeat_previous_easy", 1, 4),
        ("popgym_repeat_previous_medium", 2, 32),
        ("popgym_repeat_previous_hard", 3, 64),
    ],
)
def test_repeat_previous_variant_query_delay_and_episode_length(env_name, num_decks, k):
    episode_length = 52 * num_decks - 1
    with vec_for(env_name) as vec:
        assert vec.obs_size == 4
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [4]

        obs = obs_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        assert obs[0, 1] == 0  # query not yet available at reset (k > 1)

        first_query_step = None
        for step in range(1, episode_length + 1):
            vec.cpu_step(action.ctypes.data)
            expected_terminal = 1.0 if step == episode_length else 0.0
            assert terminals[0] == expected_terminal
            if first_query_step is None and not terminals[0] and obs[0, 1] == 1:
                first_query_step = step

        # The k-step lookback query first becomes available at tick k-1.
        assert first_query_step == k - 1
