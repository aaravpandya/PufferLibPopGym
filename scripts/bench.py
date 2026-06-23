"""Benchmark native PufferLib envs or reference Python POPGym envs.

Build an env first, for example:
    PYTHON=.venv/bin/python ./build.sh popgym_repeat_previous --cpu

Then run:
    .venv/bin/python scripts/bench.py --total-agents 8192 --steps 10000
    .venv/bin/python scripts/bench.py --backend python --python-env popgym-MineSweeperEasy-v0
"""

from __future__ import annotations

import argparse
import ctypes
from time import perf_counter
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-agents", type=int, default=8192)
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--backend", choices=["native", "python"], default="native")
    parser.add_argument(
        "--native-env",
        type=str,
        default="",
        help="Load pufferlib.native._C_<env> instead of legacy pufferlib._C.",
    )
    parser.add_argument(
        "--python-env",
        type=str,
        default="",
        help="Gymnasium/POPGym env id or shorthand for --backend python.",
    )

    parser.add_argument("--num-decks", type=int, default=1)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--include-prev-action", action="store_true")
    parser.add_argument("--include-antialias", action="store_true")
    return parser.parse_args()


def load_backend(native_env: str):
    if native_env:
        from pufferlib.native_envs import load_native_env

        return load_native_env(native_env)

    import pufferlib._C as backend

    return backend


def make_actions(rng: np.random.Generator, vec: Any, random_actions: bool) -> np.ndarray:
    shape = (vec.total_agents, vec.num_atns)
    if random_actions:
        actions = np.zeros(shape, dtype=np.float32)
        for head, high in enumerate(vec.act_sizes):
            actions[:, head] = rng.integers(
                0, int(high), size=vec.total_agents, dtype=np.int32
            )
        return actions
    return np.zeros(shape, dtype=np.float32)


def read_float_buffer(ptr: int, size: int) -> np.ndarray:
    array_type = ctypes.c_float * size
    return np.ctypeslib.as_array(array_type.from_address(ptr))


def normalize_python_env_id(env_name: str) -> str:
    aliases = {
        "popgym_minesweeper": "popgym-MineSweeperEasy-v0",
        "minesweeper": "popgym-MineSweeperEasy-v0",
        "popgym_repeat_previous": "popgym-RepeatPreviousEasy-v0",
        "repeat_previous": "popgym-RepeatPreviousEasy-v0",
    }
    return aliases.get(env_name, env_name or "popgym-RepeatPreviousEasy-v0")


def make_python_vec(env_id: str, total_agents: int):
    import gymnasium as gym
    import popgym  # noqa: F401

    return gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(total_agents)])


def make_python_actions(
    rng: np.random.Generator,
    single_space: Any,
    total_agents: int,
    random_actions: bool,
):
    import gymnasium as gym

    if isinstance(single_space, gym.spaces.Discrete):
        if random_actions:
            return rng.integers(0, single_space.n, size=(total_agents,), dtype=np.int64)
        return np.zeros((total_agents,), dtype=np.int64)

    if isinstance(single_space, gym.spaces.MultiDiscrete):
        shape = (total_agents, len(single_space.nvec))
        if random_actions:
            actions = np.zeros(shape, dtype=np.int64)
            for head, high in enumerate(single_space.nvec):
                actions[:, head] = rng.integers(0, int(high), size=total_agents, dtype=np.int64)
            return actions
        return np.zeros(shape, dtype=np.int64)

    return np.asarray([single_space.sample() for _ in range(total_agents)])


def run_native(args: argparse.Namespace, rng: np.random.Generator) -> None:
    C = load_backend(args.native_env)

    vec_args = {
        "vec": {
            "total_agents": args.total_agents,
            "num_buffers": args.num_buffers,
            "num_threads": args.num_threads,
        },
        "env": {
            "num_decks": args.num_decks,
            "k": args.k,
            "include_prev_action": int(args.include_prev_action),
            "include_antialias": int(args.include_antialias),
        },
    }

    vec = C.create_vec(vec_args, 0)
    vec.reset()
    try:
        print(f"env_name={C.env_name}")
        print(
            "vec="
            f"total_agents={vec.total_agents} "
            f"num_buffers={args.num_buffers} "
            f"num_threads={args.num_threads}"
        )
        print(
            "env="
            f"num_decks={args.num_decks} "
            f"k={args.k} "
            f"include_prev_action={int(args.include_prev_action)} "
            f"include_antialias={int(args.include_antialias)}"
        )
        print(
            "spaces="
            f"obs_size={vec.obs_size} "
            f"obs_dtype={vec.obs_dtype} "
            f"num_atns={vec.num_atns} "
            f"act_sizes={list(vec.act_sizes)}"
        )

        actions = make_actions(rng, vec, args.random_actions)
        for _ in range(args.warmup_steps):
            if args.random_actions:
                actions[:] = make_actions(rng, vec, True)
            vec.cpu_step(actions.ctypes.data)

        t0 = perf_counter()
        for _ in range(args.steps):
            if args.random_actions:
                actions[:] = make_actions(rng, vec, True)
            vec.cpu_step(actions.ctypes.data)
        elapsed = perf_counter() - t0

        agent_steps = vec.total_agents * args.steps
        rewards = read_float_buffer(vec.rewards_ptr, vec.total_agents)
        terminals = read_float_buffer(vec.terminals_ptr, vec.total_agents)

        print(f"elapsed_sec={elapsed:.6f}")
        print(f"vec_steps_per_sec={args.steps / elapsed:,.1f}")
        print(f"agent_steps_per_sec={agent_steps / elapsed:,.0f}")
        print(f"million_agent_steps_per_sec={(agent_steps / elapsed) / 1_000_000:.3f}")
        print(f"us_per_vec_step={(elapsed / args.steps) * 1_000_000:.2f}")
        print(f"reward_mean_last_step={float(rewards.mean()):.6f}")
        print(f"terminals_last_step={int(terminals.sum())}")

        logs = vec.log()
        if logs:
            rendered = " ".join(f"{key}={value:.6g}" for key, value in sorted(logs.items()))
            print(f"log {rendered}")
    finally:
        vec.close()


def run_python(args: argparse.Namespace, rng: np.random.Generator) -> None:
    env_id = normalize_python_env_id(args.python_env)
    vec = make_python_vec(env_id, args.total_agents)
    try:
        obs, _ = vec.reset(seed=args.seed)
        print(f"env_name={env_id}")
        print(
            "vec="
            f"total_agents={args.total_agents} "
            "backend=python_sync"
        )
        print(f"spaces=observation_space={vec.single_observation_space} action_space={vec.single_action_space}")

        actions = make_python_actions(
            rng, vec.single_action_space, args.total_agents, args.random_actions
        )
        rewards = np.zeros(args.total_agents, dtype=np.float32)
        terminals = np.zeros(args.total_agents, dtype=bool)
        truncations = np.zeros(args.total_agents, dtype=bool)

        for _ in range(args.warmup_steps):
            if args.random_actions:
                actions = make_python_actions(rng, vec.single_action_space, args.total_agents, True)
            obs, rewards, terminals, truncations, _ = vec.step(actions)

        t0 = perf_counter()
        for _ in range(args.steps):
            if args.random_actions:
                actions = make_python_actions(rng, vec.single_action_space, args.total_agents, True)
            obs, rewards, terminals, truncations, _ = vec.step(actions)
        elapsed = perf_counter() - t0

        agent_steps = args.total_agents * args.steps
        done = np.logical_or(terminals, truncations)

        print(f"elapsed_sec={elapsed:.6f}")
        print(f"vec_steps_per_sec={args.steps / elapsed:,.1f}")
        print(f"agent_steps_per_sec={agent_steps / elapsed:,.0f}")
        print(f"million_agent_steps_per_sec={(agent_steps / elapsed) / 1_000_000:.3f}")
        print(f"us_per_vec_step={(elapsed / args.steps) * 1_000_000:.2f}")
        print(f"reward_mean_last_step={float(np.mean(rewards)):.6f}")
        print(f"terminals_last_step={int(np.sum(terminals))}")
        print(f"truncations_last_step={int(np.sum(truncations))}")
        print(f"dones_last_step={int(np.sum(done))}")
        print(f"obs_shape={np.asarray(obs).shape} obs_dtype={np.asarray(obs).dtype}")
    finally:
        vec.close()


def run() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if args.backend == "python":
        run_python(args, rng)
    else:
        run_native(args, rng)


if __name__ == "__main__":
    run()
