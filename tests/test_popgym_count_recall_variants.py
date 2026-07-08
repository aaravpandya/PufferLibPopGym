import numpy as np
import pytest

from tests.popgym_helpers import obs_array, rewards_array, terminals_array, vec_for


@pytest.mark.parametrize(
    ("env_name", "num_values", "num_actions", "episode_length"),
    [
        ("popgym_count_recall_easy", 2, 27, 51),
        ("popgym_count_recall_medium", 4, 27, 103),
        ("popgym_count_recall_hard", 13, 17, 207),
    ],
)
def test_count_recall_variant_metadata_and_correct_count(
    env_name, num_values, num_actions, episode_length
):
    with vec_for(env_name) as vec:
        assert vec.obs_size == 2
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [num_actions]

        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        assert np.all(obs < num_values)

        counts = [0] * num_values
        counts[int(obs[0, 0])] += 1
        prev_query = int(obs[0, 1])
        action = np.array([[counts[prev_query]]], dtype=np.float32)
        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(1.0 / episode_length)
        assert terminals[0] == 0.0
        assert np.all(obs < num_values)
