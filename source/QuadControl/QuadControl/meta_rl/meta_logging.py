from __future__ import annotations

from collections import Counter

import numpy as np

from .task_sampler import QuadTask


class MetaLogger:
    """Helper around SummaryWriter for structured meta-RL logging."""

    def __init__(self, writer, enabled: bool = True):
        self.writer = writer
        self.enabled = enabled

    def log_scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        if not self.enabled:
            return
        for key, value in values.items():
            self.writer.add_scalar(f"{prefix}/{key}", float(value), step)

    def log_task(self, task: QuadTask, step: int, prefix: str = "Meta/Task") -> None:
        if not self.enabled:
            return
        self.writer.add_scalar(f"{prefix}/Family/{task.family}", 1.0, step)
        for key, value in task.params.items():
            self.writer.add_scalar(f"{prefix}/Params/{task.family}/{key}", float(value), step)

    def log_task_batch(self, tasks: list[QuadTask], step: int, prefix: str = "Meta/TaskBatch") -> None:
        if not self.enabled or not tasks:
            return

        family_counts = Counter(task.family for task in tasks)
        for family, count in family_counts.items():
            self.writer.add_scalar(f"{prefix}/FamilyCount/{family}", float(count), step)

        grouped: dict[tuple[str, str], list[float]] = {}
        for task in tasks:
            for key, value in task.params.items():
                grouped.setdefault((task.family, key), []).append(float(value))

        for (family, key), values in grouped.items():
            self.writer.add_scalar(f"{prefix}/ParamsMean/{family}/{key}", float(np.mean(values)), step)
            self.writer.add_scalar(f"{prefix}/ParamsStd/{family}/{key}", float(np.std(values)), step)
