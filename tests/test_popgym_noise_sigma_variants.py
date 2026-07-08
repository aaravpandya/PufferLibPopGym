"""Verify the compiled-in noise_sigma of the noisy cartpole/pendulum variants.

All modules share the same physics core, per-env-index rng seeding, and
rand_r consumption order, so the noiseless env and the three noisy
difficulties see identical underlying states and identical gaussian draws.
The reset-time observation deviations from the noiseless env must therefore
scale exactly 1:2:3 with the compiled-in sigmas 0.1/0.2/0.3.
"""

import numpy as np
import pytest

from tests.popgym_helpers import obs_array, vec_for


def _reset_obs(env_name, total_agents):
    with vec_for(env_name, total_agents=total_agents) as vec:
        return obs_array(vec, dtype="float32").copy()


def test_noisy_cartpole_sigma_spacing():
    total_agents = 64
    clean = _reset_obs("popgym_position_only_cartpole", total_agents)
    easy = _reset_obs("popgym_noisy_position_only_cartpole_easy", total_agents)
    medium = _reset_obs("popgym_noisy_position_only_cartpole_medium", total_agents)
    hard = _reset_obs("popgym_noisy_position_only_cartpole_hard", total_agents)

    # Obs are clipped to 2x the termination thresholds (columns: x, theta);
    # only compare components where no difficulty was clipped so the
    # deviations stay linear in sigma.
    limits = 0.999 * np.array([2.0 * 2.4, 2.0 * 12.0 * 2.0 * np.pi / 360.0])
    mask = (
        (np.abs(easy) < limits) & (np.abs(medium) < limits) & (np.abs(hard) < limits)
    )
    assert mask.sum() >= 20

    d_easy = (easy - clean)[mask]
    assert np.std(d_easy) > 0.01  # noise is actually applied
    assert np.allclose((medium - clean)[mask], 2.0 * d_easy, atol=1e-5)
    assert np.allclose((hard - clean)[mask], 3.0 * d_easy, atol=1e-5)


def test_noisy_pendulum_sigma_spacing():
    total_agents = 64
    clean = _reset_obs("popgym_position_only_pendulum", total_agents)
    easy = _reset_obs("popgym_noisy_position_only_pendulum_easy", total_agents)
    medium = _reset_obs("popgym_noisy_position_only_pendulum_medium", total_agents)
    hard = _reset_obs("popgym_noisy_position_only_pendulum_hard", total_agents)

    # Pendulum obs are clipped to [-1, 1]; only compare components where no
    # difficulty was clipped so the deviations stay linear in sigma.
    mask = (
        (np.abs(easy) < 0.999) & (np.abs(medium) < 0.999) & (np.abs(hard) < 0.999)
    )
    assert mask.sum() >= 20

    d_easy = (easy - clean)[mask]
    assert np.std(d_easy) > 0.01
    assert np.allclose((medium - clean)[mask], 2.0 * d_easy, atol=1e-5)
    assert np.allclose((hard - clean)[mask], 3.0 * d_easy, atol=1e-5)
