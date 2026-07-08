import numpy as np
import pytest

from pufferlib.native_envs import available_native_envs, load_native_env

from tests.popgym_helpers import (
    native_popgym_env_names,
    obs_array,
    rewards_array,
    terminals_array,
    vec_for,
)


NATIVE_POPGYM_ENVS = native_popgym_env_names()


def test_env_list_is_derived_from_ocean_layout():
    assert len(NATIVE_POPGYM_ENVS) >= 56
    for family in (
        "popgym_autoencode",
        "popgym_battleship",
        "popgym_higher_lower",
        "popgym_minesweeper",
        "popgym_repeat_previous",
    ):
        for suffix in ("", "_easy", "_medium", "_hard"):
            assert family + suffix in NATIVE_POPGYM_ENVS


@pytest.mark.parametrize("env_name", NATIVE_POPGYM_ENVS)
def test_load_native_env_reports_matching_name(env_name):
    try:
        module = load_native_env(env_name)
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))
    assert module.env_name == env_name


def test_load_popgym_envs_side_by_side():
    available = available_native_envs(list(NATIVE_POPGYM_ENVS))
    if len(available) < 2:
        pytest.skip("needs at least two built native popgym modules")

    modules = [load_native_env(env_name) for env_name in available]
    assert [module.env_name for module in modules] == available
    assert len({id(module) for module in modules}) == len(modules)


def test_native_minesweeper_smoke():
    with vec_for("popgym_minesweeper", total_agents=16) as vec:
        assert vec.obs_size == 1
        assert vec.num_atns == 2
        assert list(vec.act_sizes) == [4, 4]

        obs = obs_array(vec)
        rewards = rewards_array(vec)
        terminals = terminals_array(vec)
        actions = np.zeros((vec.total_agents, vec.num_atns), dtype=np.float32)

        vec.cpu_step(actions.ctypes.data)
        assert obs.shape == (vec.total_agents, vec.obs_size)
        assert obs.dtype == np.uint8
        assert np.all(obs >= 0)
        assert np.all(obs <= 2)
        assert rewards.shape == (vec.total_agents,)
        assert terminals.shape == (vec.total_agents,)
