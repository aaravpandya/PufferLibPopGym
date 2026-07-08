"""Compile-level contract checks for the native POPGym env dirs.

The GPU build (src/bindings.cu) includes the env binding and then
src/curriculum.cu, whose rollout path calls puffer_state_refresh(env), and
build.sh's --local/--fast/--web modes compile ocean/<env>/<env>.c. Neither
toolchain path runs in the default CPU test setup, so approximate both with
clang -fsyntax-only: every binding must define a callable
puffer_state_refresh, and every env dir must ship a standalone driver.
"""

import shutil
import subprocess

import pytest

from tests.popgym_helpers import REPO_ROOT, native_popgym_env_names


CLANG = shutil.which("clang")
RAYLIB_INCLUDE = next(iter(sorted(REPO_ROOT.glob("raylib-5.5_*/include"))), None)

pytestmark = pytest.mark.skipif(CLANG is None, reason="clang not available")


def _syntax_check(source_path, extra_flags, env_name):
    cmd = [
        CLANG,
        "-fsyntax-only",
        "-DPLATFORM_DESKTOP",
        *extra_flags,
        "-I", str(REPO_ROOT),
        "-I", str(REPO_ROOT / "src"),
        "-I", str(REPO_ROOT / "src" / "cpu_stubs"),
        "-I", str(REPO_ROOT / "ocean" / env_name),
        "-I", str(REPO_ROOT / "vendor"),
    ]
    if RAYLIB_INCLUDE is not None:
        cmd += ["-I", str(RAYLIB_INCLUDE)]
    cmd.append(str(source_path))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("env_name", native_popgym_env_names())
def test_binding_defines_state_refresh_hook(env_name, tmp_path):
    binding = REPO_ROOT / "ocean" / env_name / "binding.c"
    tu = tmp_path / f"{env_name}_contract.c"
    tu.write_text(
        f'#include "{binding}"\n'
        "void puffer_contract_probe(Env* env) { puffer_state_refresh(env); }\n"
    )
    _syntax_check(tu, ["-DPUFFER_PYTHON_EXTENSION"], env_name)


@pytest.mark.parametrize("env_name", native_popgym_env_names())
def test_standalone_driver_compiles(env_name):
    driver = REPO_ROOT / "ocean" / env_name / f"{env_name}.c"
    assert driver.exists(), (
        f"{driver} missing: build.sh --local/--fast/--web compile this file"
    )
    if RAYLIB_INCLUDE is None:
        pytest.skip("raylib headers not downloaded")
    _syntax_check(driver, [], env_name)
