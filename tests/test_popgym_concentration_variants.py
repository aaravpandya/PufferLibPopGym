import ctypes

import numpy as np
import pytest


def _skip_if_wrong_env(expected):
    try:
        import pufferlib._C as C
    except Exception as exc:
        pytest.skip(f"pufferlib._C unavailable or build required: {exc}")

    if C.env_name != expected:
        pytest.skip(
            "Build the target env first: "
            f"PYTHON=.venv/bin/python ./build.sh {expected} --cpu"
        )
    return C


@pytest.mark.parametrize(
    ("env_name", "num_cards", "facedown", "episode_length"),
    [
        ("popgym_concentration_easy", 52, 2, 104),
        ("popgym_concentration_medium", 104, 2, 208),
    ],
)
def test_concentration_variant_metadata_and_penalty(
    env_name, num_cards, facedown, episode_length
):
    C = _skip_if_wrong_env(env_name)
    args = {"vec": {"total_agents": 1, "num_buffers": 1, "num_threads": 1}, "env": {}}
    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        assert vec.obs_size == num_cards
        assert vec.num_atns == 1
        assert list(vec.act_sizes) == [num_cards]

        obs = np.ctypeslib.as_array((ctypes.c_ubyte * vec.obs_size).from_address(vec.obs_ptr))
        rewards = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.rewards_ptr))
        terminals = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(vec.terminals_ptr))
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
    finally:
        vec.close()
