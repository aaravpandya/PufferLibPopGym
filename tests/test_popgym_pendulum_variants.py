import numpy as np
import pytest

from tests.popgym_helpers import steps_until_terminal, vec_for


@pytest.mark.parametrize(
    ("env_name", "max_episode_length"),
    [
        ("popgym_position_only_pendulum_easy", 200),
        ("popgym_position_only_pendulum_medium", 150),
        ("popgym_position_only_pendulum_hard", 100),
        ("popgym_noisy_position_only_pendulum_easy", 200),
        ("popgym_noisy_position_only_pendulum_medium", 200),
        ("popgym_noisy_position_only_pendulum_hard", 200),
    ],
)
def test_pendulum_variant_episode_length(env_name, max_episode_length):
    # Pendulum episodes only end on timeout, so the episode length equals the
    # compiled-in max_episode_length exactly.
    with vec_for(env_name) as vec:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [1]

        action = np.zeros((1, 1), dtype=np.float32)
        assert steps_until_terminal(vec, action, max_episode_length + 1) == max_episode_length
