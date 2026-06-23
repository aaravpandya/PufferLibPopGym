"""Helpers for loading selected native env extensions.

Each selected native env can be built as its own extension module, for example:

    ./build.sh popgym_repeat_previous --cpu \
        --module-name _C_popgym_repeat_previous \
        --output-dir pufferlib/native

Then load it with:

    C = load_native_env("popgym_repeat_previous")
"""

from __future__ import annotations

import importlib
import importlib.util


def _module_name(env_name: str) -> str:
    if not env_name.isidentifier():
        raise ValueError(f"Invalid native env name: {env_name!r}")
    return f"pufferlib.native._C_{env_name}"


def load_native_env(env_name: str):
    """Import and return the native extension for ``env_name``."""

    module_name = _module_name(env_name)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ModuleNotFoundError(
            f"Native env {env_name!r} is not built. Build it with: "
            f"./build.sh {env_name} --cpu --module-name _C_{env_name} "
            "--output-dir pufferlib/native"
        ) from exc

    built_name = getattr(module, "env_name", None)
    if built_name != env_name:
        raise RuntimeError(
            f"Native module {module_name!r} reports env_name={built_name!r}, "
            f"expected {env_name!r}"
        )
    return module


def available_native_envs(env_names: list[str]) -> list[str]:
    """Return the names from ``env_names`` that have a built native module."""

    return [
        env_name
        for env_name in env_names
        if importlib.util.find_spec(_module_name(env_name)) is not None
    ]
