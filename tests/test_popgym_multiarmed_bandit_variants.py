import numpy as np
import pytest

from tests.popgym_helpers import rewards_array, terminals_array, vec_for


@pytest.mark.parametrize(
    ("env_name", "num_bandits", "episode_length"),
    [
        ("popgym_multiarmed_bandit_easy", 10, 200),
        ("popgym_multiarmed_bandit_medium", 20, 400),
        ("popgym_multiarmed_bandit_hard", 30, 600),
    ],
)
def test_multiarmed_bandit_variant_metadata_and_reward_scale(
    env_name, num_bandits, episode_length
):
    with vec_for(env_name) as vec:
        assert vec.obs_size == 1
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [num_bandits]

        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        vec.cpu_step(action.ctypes.data)
        assert abs(rewards[0]) == pytest.approx(1.0 / episode_length)
        assert terminals[0] == 0.0
