import ctypes

import numpy as np
import pytest

from pufferlib.native_envs import load_native_env


NATIVE_POPGYM_ENVS = (
    "popgym_autoencode",
    "popgym_autoencode_easy",
    "popgym_autoencode_hard",
    "popgym_autoencode_medium",
    "popgym_battleship",
    "popgym_battleship_easy",
    "popgym_battleship_hard",
    "popgym_battleship_medium",
    "popgym_concentration",
    "popgym_concentration_easy",
    "popgym_concentration_hard",
    "popgym_concentration_medium",
    "popgym_count_recall",
    "popgym_count_recall_easy",
    "popgym_count_recall_hard",
    "popgym_count_recall_medium",
    "popgym_higher_lower",
    "popgym_higher_lower_easy",
    "popgym_higher_lower_hard",
    "popgym_higher_lower_medium",
    "popgym_minesweeper",
    "popgym_minesweeper_easy",
    "popgym_minesweeper_hard",
    "popgym_minesweeper_medium",
    "popgym_multiarmed_bandit",
    "popgym_multiarmed_bandit_easy",
    "popgym_multiarmed_bandit_hard",
    "popgym_multiarmed_bandit_medium",
    "popgym_noisy_position_only_cartpole",
    "popgym_noisy_position_only_cartpole_easy",
    "popgym_noisy_position_only_cartpole_hard",
    "popgym_noisy_position_only_cartpole_medium",
    "popgym_noisy_position_only_pendulum",
    "popgym_noisy_position_only_pendulum_easy",
    "popgym_noisy_position_only_pendulum_hard",
    "popgym_noisy_position_only_pendulum_medium",
    "popgym_position_only_cartpole",
    "popgym_position_only_cartpole_easy",
    "popgym_position_only_cartpole_hard",
    "popgym_position_only_cartpole_medium",
    "popgym_position_only_pendulum",
    "popgym_position_only_pendulum_easy",
    "popgym_position_only_pendulum_hard",
    "popgym_position_only_pendulum_medium",
    "popgym_repeat_first",
    "popgym_repeat_first_easy",
    "popgym_repeat_first_hard",
    "popgym_repeat_first_medium",
    "popgym_repeat_previous",
    "popgym_repeat_previous_easy",
    "popgym_repeat_previous_hard",
    "popgym_repeat_previous_medium",
    "popgym_velocity_only_cartpole",
    "popgym_velocity_only_cartpole_easy",
    "popgym_velocity_only_cartpole_hard",
    "popgym_velocity_only_cartpole_medium",
)


def _load_or_skip(env_name):
    try:
        return load_native_env(env_name)
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))


def test_load_popgym_envs_side_by_side():
    modules = [_load_or_skip(env_name) for env_name in NATIVE_POPGYM_ENVS]

    assert [module.env_name for module in modules] == list(NATIVE_POPGYM_ENVS)
    assert len({id(module) for module in modules}) == len(modules)


def test_native_minesweeper_smoke():
    C = _load_or_skip("popgym_minesweeper")
    args = {
        "vec": {"total_agents": 16, "num_buffers": 1, "num_threads": 1},
        "env": {"difficulty": 0},
    }

    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        assert vec.obs_size == 1
        assert vec.num_atns == 2
        assert list(vec.act_sizes) == [4, 4]

        obs = np.ctypeslib.as_array(
            (ctypes.c_ubyte * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
        ).reshape(vec.total_agents, vec.obs_size)
        rewards = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
        )
        terminals = np.ctypeslib.as_array(
            (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
        )
        actions = np.zeros((vec.total_agents, vec.num_atns), dtype=np.float32)

        vec.cpu_step(actions.ctypes.data)
        assert obs.shape == (vec.total_agents, vec.obs_size)
        assert obs.dtype == np.uint8
        assert np.all(obs >= 0)
        assert np.all(obs <= 2)
        assert rewards.shape == (vec.total_agents,)
        assert terminals.shape == (vec.total_agents,)
    finally:
        vec.close()
