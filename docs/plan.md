# PufferLib POPGym Rewrite Plan

## Goal

Build a high-throughput PufferLib-native implementation of selected POPGym
environments for training memory-focused reinforcement learning agents at scale.

The purpose is not to reproduce POPGym's random number streams or exact Python
execution traces. The purpose is to preserve the behavioral task definitions
well enough that policies trained in fast C/PufferLib environments transfer to
official POPGym evaluation.

In short:

- Train fast in C/PufferLib.
- Evaluate honestly on official Python POPGym.
- Use speed to iterate on small memory models, PPO variants, and custom memory
  baselines.

## Motivation

POPGym environments are intentionally small, partially observable tasks for
testing memory. Many of them are simple enough to rewrite in C with fixed-size
arrays and direct batched stepping. PufferLib can then run many environments in
parallel with much lower Python overhead, which should make it possible to keep
even modest GPUs busy while training small recurrent, attention, or
Hopfield-style policies.

This project treats the C environments as a fast training distribution and
official POPGym as the held-out judge. The agent should learn the behavior of
the task family, not quirks of a specific RNG, deck shuffle, or maze generator.

## Core Contract

The C/PufferLib environments should match POPGym at the semantic level:

- Observation spaces and encodings should match the training harness.
- Action spaces should match, including any action flattening conventions.
- Reward signs, scales, and terminal/truncated behavior should be equivalent.
- Episode lengths and difficulty settings should be close to POPGym.
- Wrapper assumptions such as previous-action observations and antialiasing
  should be explicit.
- Policies trained in C should be regularly evaluated on official POPGym.

Exact seed parity is not required. Different random sequences are acceptable as
long as the task distribution remains comparable and transfer works.

## Non-Goals

- Do not claim to replace official POPGym.
- Do not require bit-for-bit equivalence with NumPy RNG or Python traces.
- Do not vendor large chunks of POPGym Python source.
- Do not optimize for one fixed generated sequence, maze, or board layout.
- Do not make the C environments easier in ways that break transfer.

## Initial Environment Targets

Start with environments that are simple, discrete, and memory-relevant:

1. `CountRecall`
2. `RepeatPrevious`
3. `RepeatFirst`
4. `Autoencode`
5. `MineSweeper`
6. `HigherLower`
7. `MultiarmedBandit`

Defer more delicate environments:

- `Labyrinth*`: useful later, but maze generation and task distribution need
  more care.
- CartPole/Pendulum variants: exact physics fidelity is less central to the
  memory baseline goal and easier to get subtly wrong.
- `Concentration` and `Battleship`: good candidates after the first batch, but
  they have more state bookkeeping.

## Validation Strategy

Use official POPGym as the reference evaluator, but avoid overfitting the C
rewrite to exact Python traces.

Recommended checks:

- Space checks: observation/action shapes, bounds, dtypes, and flattening.
- Invariant checks: rewards stay within expected ranges, done flags occur at
  valid times, resets produce valid states.
- Hand-authored edge cases: known actions in controlled states produce expected
  rewards and transitions.
- Random-policy statistics: compare return and episode-length distributions
  against POPGym.
- Transfer checks: train in C/PufferLib, evaluate checkpoints in official
  POPGym throughout training.

If transfer fails, treat it as evidence of a semantic mismatch or training issue,
not as evidence that RNG parity is required.

## Training Architecture

The intended setup is:

- `train_backend = puffer_c`
- `eval_backend = popgym`
- PPO or PPO-like training loop with batched PufferLib rollouts.
- Small neural policies: MLP, GRU/LSTM, attention, and Hopfield-style memory.
- Frequent checkpoint evaluation on official POPGym.

The C backend should be optimized for throughput:

- Fixed-size env state structs.
- Batched in-place stepping.
- Direct observation, reward, terminal, and truncation buffers.
- Minimal allocation during step.
- Fast RNG that is good enough for diverse task generation.

## Success Criteria

The rewrite is successful if:

- C/PufferLib envs deliver substantially higher environment throughput than the
  current Python vectorizer.
- Policies trained on the C environments transfer to official POPGym evaluation.
- The training loop can scale to large batches and keep small GPU workloads fed.
- The project remains easy to reason about: official POPGym is the judge, and
  C/PufferLib is the speed engine.

## Repository Positioning

Suggested repo names:

- `pufferlib-popgym-envs`
- `pufferlib-popgym`
- `popgym-puffer`
- `popgym-native-puffer`

Suggested description:

> Experimental POPGym environment ports and memory-baseline training code for
> PufferLib. Not affiliated with or endorsed by PufferAI.

## Guiding Principle

We are not learning the RNG. We are learning the behavior.

The policy should solve the task regardless of which sequence, deck, board, or
maze instance is generated. The rewrite should therefore prioritize fast,
diverse, semantically faithful task generation over exact replay of POPGym's
Python internals.
