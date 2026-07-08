"""Shared helpers for the native POPGym env tests."""

import ctypes
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from pufferlib.native_envs import load_native_env

REPO_ROOT = Path(__file__).resolve().parents[1]

_CTYPES = {
    "uint8": ctypes.c_ubyte,
    "float32": ctypes.c_float,
}


def native_popgym_env_names():
    """All popgym env dirs under ocean/, sorted.

    Requiring binding.c keeps build artifacts (e.g. a stray .dSYM directory)
    out of the env list.
    """
    names = tuple(sorted(
        path.name
        for path in (REPO_ROOT / "ocean").glob("popgym_*")
        if path.is_dir() and (path / "binding.c").is_file()
    ))
    assert "popgym_higher_lower" in names, "unexpected ocean/ layout"
    return names


def get_env_module(env_name):
    """Return a built native module for ``env_name``, or skip the test.

    Prefers the side-by-side ``pufferlib.native._C_<env>`` module (the batch
    build the suite documents), falling back to the single-slot
    ``pufferlib._C`` build so the build-one-env workflow still works. The
    fallback order matters: a stale ``pufferlib._C`` can be linked against a
    second OpenMP runtime, which aborts the process when torch is loaded.
    """
    try:
        return load_native_env(env_name)
    except ModuleNotFoundError:
        pass
    try:
        import pufferlib._C as C
        if getattr(C, "env_name", None) == env_name:
            return C
    except Exception:
        pass
    # PUFFER_NO_OPENMP keeps the module loadable next to torch, which
    # bundles its own libomp (loading two OpenMP runtimes aborts).
    pytest.skip(
        "Build the target env first: "
        f"PUFFER_NO_OPENMP=1 PYTHON=.venv/bin/python ./build.sh {env_name} "
        f"--cpu --module-name _C_{env_name} --output-dir pufferlib/native"
    )


@contextmanager
def vec_for(env_name, *, total_agents=1, num_threads=1, env_kwargs=None):
    C = get_env_module(env_name)
    args = {
        "vec": {
            "total_agents": total_agents,
            "num_buffers": 1,
            "num_threads": num_threads,
        },
        "env": dict(env_kwargs or {}),
    }
    # create_vec's second argument is the gpu flag, not a seed; env rngs are
    # seeded deterministically by env index in my_vec_init.
    vec = C.create_vec(args, 0)
    vec.reset()
    try:
        yield vec
    finally:
        vec.close()


def obs_array(vec, dtype="uint8"):
    ctype = _CTYPES[np.dtype(dtype).name]
    arr = np.ctypeslib.as_array(
        (ctype * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
    )
    return arr.reshape(vec.total_agents, vec.obs_size)


def rewards_array(vec):
    return np.ctypeslib.as_array(
        (ctypes.c_float * vec.total_agents).from_address(vec.rewards_ptr)
    )


def terminals_array(vec):
    return np.ctypeslib.as_array(
        (ctypes.c_float * vec.total_agents).from_address(vec.terminals_ptr)
    )


def steps_until_terminal(vec, actions, limit):
    """Step env 0 with ``actions`` until it terminates; return the step count."""
    terminals = terminals_array(vec)
    for step in range(1, limit + 1):
        vec.cpu_step(actions.ctypes.data)
        if terminals[0]:
            return step
    pytest.fail(f"episode did not terminate within {limit} steps")
