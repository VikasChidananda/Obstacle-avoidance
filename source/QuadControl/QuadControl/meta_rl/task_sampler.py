from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuadTask:
    """Container for a sampled quadrotor tracking task."""

    family: str
    params: dict[str, float]


_DEFAULT_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "hover": {
        "x": (-1.5, 1.5),
        "y": (-1.5, 1.5),
        "height": (0.6, 1.8),
    },
    "straight": {
        "speed": (0.1, 0.6),
        "x_dir": (-1.0, 1.0),
        "y_dir": (-1.0, 1.0),
        "height": (0.6, 1.8),
    },
    "circle": {
        "radius": (0.8, 2.5),
        "speed": (0.08, 0.4),
        "height": (0.0, 0.7),
        "z_center": (0.8, 1.5),
    },
    "lemniscate": {
        "a": (0.8, 2.2),
        "speed": (0.08, 0.35),
        "z_center": (0.8, 1.5),
    },
}


class QuadTaskSampler:
    """Sample families and parameters for meta-RL quadrotor tasks."""

    def __init__(
        self,
        families: list[str] | tuple[str, ...] | None = None,
        seed: int | None = None,
        ranges: dict[str, dict[str, tuple[float, float]]] | None = None,
    ) -> None:
        self._ranges = ranges or _DEFAULT_RANGES
        self._families = list(families or self._ranges.keys())
        if not self._families:
            raise ValueError("Task sampler requires at least one task family")
        for family in self._families:
            if family not in self._ranges:
                raise ValueError(f"Unsupported task family: {family}")
        self._rng = np.random.default_rng(seed)

    @property
    def families(self) -> list[str]:
        return list(self._families)

    def sample_task(self, family: str | None = None) -> QuadTask:
        task_family = family or self._rng.choice(self._families).item()
        if task_family not in self._ranges:
            raise ValueError(f"Unsupported task family: {task_family}")

        params = {
            name: float(self._rng.uniform(low, high))
            for name, (low, high) in self._ranges[task_family].items()
        }

        if task_family == "straight":
            # Keep direction meaningful and avoid a near-zero vector.
            params["x_dir"] = 1.0 if params["x_dir"] >= 0.0 else -1.0
            params["y_dir"] = 1.0 if params["y_dir"] >= 0.0 else -1.0

        return QuadTask(family=task_family, params=params)

    def sample_like(self, task: QuadTask) -> QuadTask:
        """Sample a new task from the same family as a support task."""
        return self.sample_task(family=task.family)

    def sample_batch(self, count: int, families: list[str] | None = None) -> list[QuadTask]:
        if count <= 0:
            return []

        if families is None:
            return [self.sample_task() for _ in range(count)]

        if not families:
            raise ValueError("families cannot be empty when provided")

        return [self.sample_task(family=families[idx % len(families)]) for idx in range(count)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "families": self.families,
            "ranges": self._ranges,
        }
