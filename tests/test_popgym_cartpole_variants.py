import numpy as np
import pytest

from tests.popgym_helpers import rewards_array, terminals_array, vec_for


@pytest.mark.parametrize(
    ("env_name", "max_episode_length"),
    [
        ("popgym_position_only_cartpole_easy", 200),
        ("popgym_position_only_cartpole_medium", 400),
        ("popgym_position_only_cartpole_hard", 600),
        ("popgym_velocity_only_cartpole_easy", 200),
        ("popgym_velocity_only_cartpole_medium", 400),
        ("popgym_velocity_only_cartpole_hard", 600),
        ("popgym_noisy_position_only_cartpole_easy", 200),
        ("popgym_noisy_position_only_cartpole_medium", 200),
        ("popgym_noisy_position_only_cartpole_hard", 200),
    ],
)
def test_cartpole_variant_reward_scale(env_name, max_episode_length):
    # Per-step reward is 1/max_episode_length, so a single step pins down the
    # compiled-in episode length.
    with vec_for(env_name) as vec:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [2]

        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(1.0 / max_episode_length)
        assert terminals[0] == 0.0
