"""Build selected native envs as side-by-side importable extensions."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("envs", nargs="+", help="Env names under ocean/, e.g. popgym_minesweeper")
    parser.add_argument("--cpu", action="store_true", help="Build CPU extensions")
    parser.add_argument("--output-dir", default="pufferlib/native")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    mode = "--cpu" if args.cpu else "--float"

    env = os.environ.copy()
    env["PYTHON"] = args.python

    for env_name in args.envs:
        module_name = f"_C_{env_name}"
        cmd = [
            str(root / "build.sh"),
            env_name,
            mode,
            "--module-name",
            module_name,
            "--output-dir",
            args.output_dir,
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
