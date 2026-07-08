# Repository Guidelines

## Project Structure & Module Organization
PufferLib combines a Python API with native C/CUDA environments. Core Python modules live in `pufferlib/`; shared CUDA, C++, and pybind code lives in `src/`. Native environments are organized under `ocean/<env>/`, usually as `<env>.c`, `<env>.h`, and `binding.c`; use `ocean/template/` as the minimal reference. Runtime presets live in `config/*.ini`, assets in `resources/<env>/`, examples in `examples/`, and regression/parity tests in `tests/`.

## Build, Test, and Development Commands
- `python -m pip install -e . --no-build-isolation`: install the package in editable mode, matching the GitHub Actions install path.
- `./build.sh breakout`: build `pufferlib/_C*.so` with the selected environment statically linked; requires CUDA tooling for the default GPU backend.
- `./build.sh breakout --cpu`: build the CPU fallback backend.
- `./build.sh breakout --local` or `./build.sh breakout --fast`: build a standalone executable for manual environment debugging.
- `pytest tests/test_api.py`: run a focused pytest file; use `pytest tests/` for the broader suite when dependencies and native builds are available.
- `python examples/puffer_env.py`: smoke-test the basic Python environment interface.

## Coding Style & Naming Conventions
Use 4-space indentation in Python and C/CUDA. Follow local naming: Python modules, config files, and environment directories use `snake_case`; C structs and typedef-style environment names may use `PascalCase` when existing code does. Keep each native environment self-contained in its `ocean/<env>/` directory and keep matching assets/configs named after the environment. Prefer small, direct helpers over broad abstractions unless a nearby module already establishes the pattern.

## Testing Guidelines
Tests are pytest-oriented and generally named `tests/test_*.py`; CUDA/native checks may use `.cu` files or script-style parity tests. Add focused tests next to related coverage, especially for API behavior, model kernels, and environment parity. When changing a native environment, build that environment first, then run its focused tests or parity script before the full suite.

## Commit & Pull Request Guidelines
Recent history uses short imperative or descriptive subjects, for example `Fix continuous action logprob precision mismatch` and `Add NetHack environment (native C vecenv)`. Keep commits scoped and mention the affected environment/module when useful. Pull requests should describe behavior changes, list build/test commands run, link issues when applicable, and include screenshots or recordings for rendering/UI changes.

## Generated Files & Local State
Do not commit generated outputs or local experiment state. `.gitignore` already excludes `build/`, `dist/`, `*.so`, `*.o`, `raylib*/`, `box2d*/`, `checkpoints/`, `experiments/`, and `wandb/`. If `build.sh` downloads native dependencies, leave them local.
