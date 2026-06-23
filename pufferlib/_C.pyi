from __future__ import annotations

from typing import Any

precision_bytes: int

env_name: str
gpu: int


def create_vec(args: dict[str, dict[str, object]], gpu: int = 0) -> "VecEnv": ...

def puff_advantage_cpu(*args: Any, **kwargs: Any) -> Any: ...


class VecEnv:
    total_agents: int
    obs_size: int
    num_atns: int
    act_sizes: list[int]
    obs_dtype: str
    obs_elem_size: int

    @property
    def gpu(self) -> int: ...

    @property
    def obs_ptr(self) -> int: ...

    @property
    def rewards_ptr(self) -> int: ...

    @property
    def terminals_ptr(self) -> int: ...

    def reset(self) -> None: ...
    def cpu_step(self, actions_ptr: int) -> None: ...
    def render(self, env_id: int) -> None: ...
    def log(self) -> dict[str, Any]: ...
    def close(self) -> None: ...
