# POPGym on PufferLib Implementation Plan

## Purpose

Build simple, high-throughput C implementations of selected POPGym environments
inside this PufferLib fork, with an emphasis on CPU-first development on Apple
Silicon and later Linux/CUDA scaling if needed.

The practical target is not a perfect clone of POPGym internals. The target is:

- train quickly on native C/PufferLib environments,
- validate task semantics against official POPGym,
- evaluate trained policies on official POPGym,
- package the fork so training code can depend on it reproducibly.

This plan assumes the `popgym-5-cpu` branch, based on upstream PufferLib `5.0`
plus the local CPU build fixes.

## Design Principles

Use official POPGym as the judge, not as code to vendor. POPGym documents the
environment collection, wrappers such as `PreviousAction`, `Antialias`,
`Flatten`, and `DiscreteAction`, and the fact that environments follow the
Gymnasium API. The C ports should match the chosen wrapped observation/action
contract and reward/termination semantics, but exact RNG parity is not required.

Start with one very small environment, make the whole loop work, then copy the
pattern. The recommended first target is `RepeatPrevious`, followed by
`RepeatFirst`, `Autoencode`, `MultiarmedBandit`, `HigherLower`, `CountRecall`,
and finally `MineSweeper`.

Avoid stale PufferLib templates. In this branch, use the modern `vecenv.h`
pattern. Good templates are:

- `ocean/password/`: best simple memory-task template.
- `ocean/minimal/`: compact multi-agent layout.
- `ocean/g2048/`: richer `State` plus `ByteTensor` example.
- `ocean/maze/binding.c`: custom vector init/close example.

Do not copy `ocean/template/` or older envs that include `../env_binding.h`.

## Repository Baseline

Before implementing POPGym ports, keep the branch in a known-good state:

```bash
git switch popgym-5-cpu
uv sync
./build.sh breakout --cpu
python -c "import pufferlib._C as C; print(C.env_name, C.gpu)"
```

Expected:

```text
breakout 0
```

Commit the generic CPU build support separately:

```bash
git add build.sh src/cpu_stubs/cuda_runtime_api.h
git commit -m "Support CPU builds without CUDA headers on macOS"
```

## PufferLib Native Env Contract

Each native env is compiled into the single extension module
`pufferlib/_C*.so`. Only one env is linked at a time. Build before use:

```bash
PYTHON=.venv/bin/python ./build.sh popgym_repeat_previous --cpu
```

The extension exposes the compiled env name:

```bash
python -c "import pufferlib._C as C; print(C.env_name, C.gpu)"
```

For a new env named `popgym_repeat_previous`, create:

```text
ocean/popgym_repeat_previous/
  popgym_repeat_previous.h
  popgym_repeat_previous.c
  binding.c
config/popgym_repeat_previous.ini
tests/test_popgym_repeat_previous.py
benchmarks/bench_popgym_env.py
```

Optional later:

```text
pufferlib_popgym/
  __init__.py
  registry.py
  wrappers.py
```

The env header must define these pieces:

- `Log`: floats only, including `float n`.
- `State`: fixed-size, copyable, no owning pointers.
- `Env`: contains `Log log`, observation pointer, `float* actions`,
  `float* rewards`, `float* terminals`, `int num_agents`, `State state`,
  and usually `unsigned int rng`.
- `c_reset(Env*)`
- `c_step(Env*)`
- `c_render(Env*)`
- `c_close(Env*)`

The binding file must define metadata before including `vecenv.h`:

```c
#include "popgym_repeat_previous.h"

#define OBS_SIZE ...
#define NUM_ATNS 1
#define ACT_SIZES {4}
#define OBS_TENSOR_T ByteTensor

#define Env RepeatPrevious
// The GPU state curriculum (src/curriculum.cu) calls this hook by name after
// restoring env->state; it must be an ordinary function, not a macro.
static inline void puffer_state_refresh(Env* env) { refresh_observations(env); }
#include "vecenv.h"
#include "../popgym_kwargs.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    // kwarg_or falls back to the base POPGym default when a kwarg is missing
    // instead of crashing on dict_get's NULL.
    env->num_decks = (int)kwarg_or(kwargs, "num_decks", 1);
    env->k = (int)kwarg_or(kwargs, "k", 4);
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "accuracy", log->accuracy);
}
```

Use `ByteTensor` for small discrete observations, `FloatTensor` for continuous
features. Actions always arrive as `float*`; cast with `(int)env->actions[0]`.

## First Env: RepeatPrevious

POPGym describes `RepeatPrevious` as a card-suit memory task: the agent sees a
sequence of cards and must output the suit from `k` timesteps ago. Start with a
semantic version:

- Four suits: `0..3`.
- Optional ranks if the observation includes full card identity.
- `num_decks` controls episode length.
- `k` controls memory distance.
- Reward `1.0` for correct suit after enough history exists, otherwise `0.0`
  or the official POPGym equivalent once verified.
- Terminal at end of deck/episode.

Choose one explicit training observation contract and write it down in the
config/test:

1. Minimal C contract: observation is current suit plus an initial/valid flag.
2. POPGym-wrapper contract: match `Flatten(Antialias(PreviousAction(env)))`.

For the first implementation, prefer the minimal C contract so the native path
is easy to debug. Then add a reference-wrapper comparison mode.

Example state shape:

```c
#define MAX_CARDS 256

typedef struct State {
    int tick;
    int episode_length;
    int history[MAX_CARDS];
    float episode_return;
} State;
```

Example observation:

```text
obs[0] = current_suit          // 0..3
obs[1] = has_query             // 0 before k steps, 1 otherwise
obs[2] = previous_action       // optional wrapper-compatible value
obs[3] = antialias_flag        // optional wrapper-compatible value
```

Keep the implementation deterministic under `env->rng`, but do not try to match
NumPy bit-for-bit.

## Config File

Create `config/popgym_repeat_previous.ini`:

```ini
[base]
env_name = popgym_repeat_previous

[vec]
total_agents = 8192
num_buffers = 1
num_threads = 4

[env]
num_decks = 1
k = 4
include_prev_action = 0
include_antialias = 0

[policy]
hidden_size = 64
num_layers = 1
expansion_factor = 1

[train]
gpus = 1
total_timesteps = 10000000
minibatch_size = 8192
horizon = 64
```

Keep integer fields as integer literals. Avoid tuned float values for fields
that the code treats as counts, such as `num_buffers`, `num_layers`,
`total_agents`, `horizon`, and `k`.

## Standalone C Runner

Create `ocean/popgym_repeat_previous/popgym_repeat_previous.c` so the env can be
run without Python:

```c
#include "popgym_repeat_previous.h"

int main(void) {
    RepeatPrevious env = {0};
    env.num_agents = 1;
    env.num_decks = 1;
    env.k = 4;
    init(&env);
    env.observations = calloc(OBS_SIZE, sizeof(unsigned char));
    env.actions = calloc(1, sizeof(float));
    env.rewards = calloc(1, sizeof(float));
    env.terminals = calloc(1, sizeof(float));

    c_reset(&env);
    for (int i = 0; i < 1000000; i++) {
        env.actions[0] = rand_r(&env.rng) % 4;
        c_step(&env);
    }
    c_close(&env);
}
```

Build it for debugging:

```bash
./build.sh popgym_repeat_previous --local
./popgym_repeat_previous
```

## Native Build and Smoke Test

Build the Python extension:

```bash
PYTHON=.venv/bin/python ./build.sh popgym_repeat_previous --cpu
```

Smoke test:

```python
import numpy as np
import pufferlib._C as C

assert C.env_name == "popgym_repeat_previous"

args = {
    "vec": {"total_agents": 16, "num_buffers": 1, "num_threads": 1},
    "env": {"num_decks": 1, "k": 4, "include_prev_action": 0, "include_antialias": 0},
}
vec = C.create_vec(args, 0)
vec.reset()

obs = np.ctypeslib.as_array(
    (np.ctypeslib.ctypes.c_uint8 * (vec.total_agents * vec.obs_size)).from_address(vec.obs_ptr)
).reshape(vec.total_agents, vec.obs_size)

actions = np.zeros((vec.total_agents, vec.num_atns), dtype=np.float32)
vec.cpu_step(actions.ctypes.data)
vec.close()
```

In the real test, use `ctypes` directly for clarity; `np.ctypeslib.ctypes` above
is only shorthand for the idea.

## Test Strategy

Add `tests/test_popgym_repeat_previous.py` with these layers:

1. Build/import guard:
   - If `pufferlib._C.env_name != "popgym_repeat_previous"`, skip with a clear
     message telling the user to run `./build.sh popgym_repeat_previous --cpu`.

2. Shape checks:
   - `vec.obs_size == expected`
   - `vec.num_atns == 1`
   - `list(vec.act_sizes) == [4]`
   - observation dtype matches `ByteTensor` expectations.

3. Hand-authored transitions:
   - Reset with a fixed seed/config.
   - Force a known suit sequence if you expose a test-only config flag or helper.
   - Step actions that should be correct/incorrect.
   - Assert reward, terminal, and observation updates.

4. Random-policy invariants:
   - Rewards stay in expected range.
   - Terminals occur only at valid episode boundaries.
   - Episode length equals expected deck length.
   - No observations exceed documented bounds.

5. Official POPGym comparison:
   - Install `popgym` as a dev dependency.
   - Construct official `RepeatPrevious` with the same difficulty.
   - Apply the chosen wrappers.
   - Run many random actions.
   - Compare distribution summaries, not exact per-seed traces:
     mean return, return standard deviation, episode length, terminal count,
     observation bounds.

Use `tests/craftax_parity.py` as the pointer-wrapping reference. It creates
`_C.create_vec`, views `obs_ptr`, `rewards_ptr`, and `terminals_ptr`, steps a
float32 action buffer, and prints high-signal divergence context.

## Benchmark Strategy

Create `benchmarks/bench_popgym_env.py` with a CLI:

```bash
python benchmarks/bench_popgym_env.py \
  --env popgym_repeat_previous \
  --agents 8192 \
  --steps 10000 \
  --threads 4
```

Benchmark both native C and official POPGym:

- Native:
  - Build target env first.
  - Create one native vec.
  - Allocate one `(agents, num_atns)` float32 action buffer.
  - Warm up for 100 steps.
  - Time `vec.cpu_step(actions.ctypes.data)` for `N` steps.
  - Report agent-steps/s and env-steps/s.

- Official POPGym:
  - Run `N` independent Gymnasium envs in a Python loop or vector wrapper.
  - Use the same random-action distribution.
  - Report env-steps/s.

Output example:

```text
env=popgym_repeat_previous agents=8192 steps=10000 threads=4
native_agent_steps_per_sec=38.2M
official_env_steps_per_sec=140k
speedup=272x
```

Save JSON output under `benchmarks/results/` only if that directory is ignored
or intentionally committed.

## Training Strategy

Do not make full PufferLib training the first milestone. The first milestone is
fast and correct environment stepping.

After native env tests and benchmarks pass, try:

```bash
python -m pufferlib.pufferl train popgym_repeat_previous --slowly
```

CPU builds expose vector helpers, while CUDA builds expose the full native
training backend. On macOS, `--slowly` uses the PyTorch backend around the C vec
env. If the upstream training CLI remains brittle on macOS, create a small
project-local training loop that uses:

- `pufferlib._C.create_vec`
- zero-copy obs/reward/terminal buffers
- `vec.cpu_step(actions_ptr)`
- a PyTorch recurrent policy
- PPO or REINFORCE baseline

Keep the training loop in `training/` or `examples/` rather than burying it in
the env implementation.

## Transfer Evaluation

Create `eval/eval_official_popgym.py`:

1. Load a checkpoint trained on the C env.
2. Construct the official POPGym env.
3. Apply the exact wrappers used in the C training contract.
4. Run deterministic evaluation episodes.
5. Report mean return, success/accuracy, and episode length.

The key success criterion is transfer:

```text
C-trained policy performs well on official POPGym with the same wrapper contract.
```

If transfer fails, debug in this order:

1. Observation encoding mismatch.
2. Action flattening mismatch.
3. Reward scale/sign mismatch.
4. Terminal/truncation mismatch.
5. Episode length/difficulty mismatch.
6. Only then investigate RNG/task distribution.

## Packaging and Shipping

There are two practical shipping modes.

### Mode A: Source Package for Your Own Training

This is the recommended first shipping path.

Use your fork as the package:

```bash
git clone git@github.com:aaravpandya/PufferLibPopGym.git
cd PufferLibPopGym
uv sync
PYTHON=.venv/bin/python ./build.sh popgym_repeat_previous --cpu
python -c "import pufferlib._C as C; print(C.env_name)"
```

Training code can depend on the repo:

```bash
pip install -e /path/to/PufferLibPopGym
```

Before running a specific env, compile that env. This is simple and matches the
current PufferLib one-env-per-extension design.

Document this explicitly:

```text
This package ships source envs. Compile the target env with build.sh before use.
The compiled pufferlib._C module contains one env at a time.
```

### Mode B: Real Wheel Later

Do not start here. A proper wheel requires more design because current
`pyproject.toml` only packages `pufferlib*`; it does not include `src/`,
`ocean/`, `config/`, `resources/`, `vendor/`, or a native build hook.

When ready:

1. Rename the distribution to avoid colliding with upstream:
   - `pufferlib-popgym`
   - or `popgym-puffer`

2. Add package data:
   - `src/**`
   - `vendor/**`
   - `ocean/popgym_*/*`
   - `config/default.ini`
   - `config/popgym_*.ini`
   - any `resources/popgym_*`

3. Decide wheel strategy:
   - one wheel that compiles one default env at install time,
   - one wheel per env,
   - or redesign `_C` to support multiple env modules.

4. Move resource/config lookup away from repo-relative paths, using
   `importlib.resources` or a package-level data directory.

5. Add a build command:
   ```bash
   python -m build
   pip install dist/pufferlib_popgym-*.whl
   ```

For now, source-install is faster, clearer, and less fragile.

## Implementation Phases

### Phase 0: Stabilize Base

- Commit the 5.0 CPU build patch.
- Verify `breakout --cpu` builds and imports.
- Add a short `docs/mac_cpu_setup.md` if needed.

Done when:

```bash
./build.sh breakout --cpu
python -c "import pufferlib._C as C; print(C.env_name, C.gpu)"
```

prints `breakout 0`.

### Phase 1: First Native POPGym Env

- Implement `ocean/popgym_repeat_previous/`.
- Add `config/popgym_repeat_previous.ini`.
- Add build/import smoke test.
- Add hand-authored transition tests.
- Add native throughput benchmark.

Done when:

```bash
./build.sh popgym_repeat_previous --cpu
pytest tests/test_popgym_repeat_previous.py
python benchmarks/bench_popgym_env.py --env popgym_repeat_previous
```

passes and reports a clear speedup.

### Phase 2: Official POPGym Validation

- Add `popgym` as a dev/test dependency.
- Add random-policy distribution comparison.
- Document the exact wrapper contract.
- Add `eval/eval_official_popgym.py`.

Done when random-policy stats are close enough and any mismatch is documented.

### Phase 3: Training Loop

- Try PufferLib `--slowly` training.
- If unstable on macOS, create a small custom PyTorch PPO/recurrent loop using
  `_C.create_vec`.
- Save checkpoints in a format the official evaluator can load.

Done when a simple recurrent model improves on the C env.

### Phase 4: Transfer

- Evaluate C-trained checkpoints on official POPGym.
- Track transfer metrics in JSON/CSV.
- Iterate only on semantic mismatches that affect transfer.

Done when C-trained policies perform meaningfully above random on official
POPGym.

### Phase 5: Add More Envs

Recommended order:

1. `popgym_repeat_first`
2. `popgym_autoencode`
3. `popgym_multiarmed_bandit`
4. `popgym_higher_lower`
5. `popgym_count_recall`
6. `popgym_minesweeper`

For each env, copy the same checklist: build, smoke, hand tests, random stats,
benchmark, transfer.

## Definition of Done for a New Env

A POPGym C port is done when:

- `./build.sh <env> --cpu` succeeds on macOS.
- `import pufferlib._C` reports the correct env name.
- Shape/action metadata matches the config.
- Hand-authored transitions pass.
- Random-policy invariant tests pass.
- Random-policy distribution is comparable to official POPGym.
- Native throughput benchmark is recorded.
- The wrapper/encoding contract is documented.
- A training script can create the native vec env.
- An evaluation script can run the trained policy on official POPGym.

## References

- POPGym repository and README: https://github.com/proroklab/popgym
- POPGym environment quickstart: https://popgym.readthedocs.io/en/latest/environment_quickstart.html
- POPGym environment API list: https://popgym.readthedocs.io/en/latest/autoapi/popgym/envs/index.html
- RepeatPrevious docs: https://popgym.readthedocs.io/en/latest/autoapi/popgym/envs/repeat_previous/index.html
- RepeatFirst docs: https://popgym.readthedocs.io/en/latest/autoapi/popgym/envs/repeat_first/index.html
- Local Puffer templates: `ocean/password/`, `ocean/minimal/`, `ocean/g2048/`
- Local pointer/parity test pattern: `tests/craftax_parity.py`
