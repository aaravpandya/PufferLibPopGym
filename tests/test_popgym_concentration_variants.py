import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


@pytest.mark.parametrize(
    ("env_name", "num_cards", "facedown", "episode_length"),
    [
        ("popgym_concentration_easy", 52, 2, 104),
        ("popgym_concentration_medium", 104, 2, 208),
        ("popgym_concentration_hard", 52, 13, 104),
    ],
)
def test_concentration_variant_metadata_and_penalty(
    env_name, num_cards, facedown, episode_length
):
    with vec_for(env_name) as vec:
        assert vec.obs_size == num_cards
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [num_cards]

        obs = obs_array(vec)[0]
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)

        assert np.all(obs == facedown)
        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == 0.0
        assert terminals[0] == 0.0
        assert obs[0] < facedown
        assert np.all(obs[1:] == facedown)

        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(-2.0 / episode_length)
        assert terminals[0] == 0.0
