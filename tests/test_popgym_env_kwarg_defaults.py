"""Base popgym envs must fall back to POPGym defaults when env kwargs are
missing (previously an unchecked dict_get NULL dereference killed the
interpreter), and must fail loudly on invalid config even in release builds.
"""

import subprocess
import sys

import numpy as np
import pytest

from tests.popgym_helpers import (
    REPO_ROOT,
    get_env_module,
    rewards_array,
    steps_until_terminal,
    vec_for,
)


@pytest.mark.parametrize(
    ("env_name", "episode_length"),
    [
        ("popgym_higher_lower", 51),
        ("popgym_autoencode", 103),
        ("popgym_repeat_first", 51),
        ("popgym_repeat_previous", 51),
        ("popgym_position_only_pendulum", 200),
        ("popgym_noisy_position_only_pendulum", 200),
    ],
)
def test_missing_env_kwargs_fall_back_to_defaults(env_name, episode_length):
    with vec_for(env_name) as vec:  # env kwargs intentionally empty
        action = np.zeros((1, 1), dtype=np.float32)
        assert steps_until_terminal(vec, action, episode_length + 1) == episode_length


@pytest.mark.parametrize(
    "env_name",
    [
        "popgym_position_only_cartpole",
        "popgym_noisy_position_only_cartpole",
        "popgym_velocity_only_cartpole",
    ],
)
def test_missing_cartpole_kwargs_default_to_200_steps(env_name):
    with vec_for(env_name) as vec:  # env kwargs intentionally empty
        rewards = rewards_array(vec)
        action = np.zeros((1, 1), dtype=np.float32)
        vec.cpu_step(action.ctypes.data)
        assert rewards[0] == pytest.approx(1.0 / 200.0)


def _create_vec_in_subprocess(env_kwargs):
    code = (
        "from pufferlib.native_envs import load_native_env\n"
        "C = load_native_env('popgym_higher_lower')\n"
        "C.create_vec({'vec': {'total_agents': 1, 'num_buffers': 1,"
        f" 'num_threads': 1}}, 'env': {env_kwargs!r}}}, 0)\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


def test_invalid_config_aborts_with_message():
    get_env_module("popgym_higher_lower")  # skip when not built
    result = _create_vec_in_subprocess({"num_decks": 0})
    assert result.returncode != 0
    assert b"num_decks must be >= 1" in result.stderr


def test_unknown_env_kwarg_aborts_with_message():
    # A misspelled kwarg must fail loudly, not silently fall back to the
    # default configuration.
    get_env_module("popgym_higher_lower")  # skip when not built
    result = _create_vec_in_subprocess({"num_deck": 1})
    assert result.returncode != 0
    assert b"unknown env kwarg: num_deck" in result.stderr
