#!/usr/bin/env python3
"""Analyze episode return by initial goal distance bins and plot histograms."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


BINS = {
    "easy": (0.1, 1.0),
    "medium": (1.0, 2.0),
    "hard": (2.0, 4.0),
}


def load_scalars(logdir: Path, tag: str) -> dict[int, float]:
    acc = event_accumulator.EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        raise KeyError(f"Tag not found: {tag}")
    return {event.step: event.value for event in acc.Scalars(tag)}


def load_first_available(logdir: Path, tags: list[str]) -> dict[int, float]:
    last_error = None
    for tag in tags:
        try:
            return load_scalars(logdir, tag)
        except KeyError as exc:
            last_error = exc
    raise SystemExit(f"Missing TensorBoard scalars. Last error: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot return histograms by initial goal distance bins.")
    parser.add_argument("--logdir", type=Path, help="TensorBoard log directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("goal_difficulty_hist.png"),
        help="Output plot path",
    )
    args = parser.parse_args()

    if not args.logdir.exists():
        raise SystemExit(f"Log directory not found: {args.logdir}")

    returns = load_first_available(
        args.logdir,
        [
            "Episode_Return/total",
            "Info / Episode_Return/total",
            "Episode_Reward/total",
            "Info / Episode_Reward/total",
        ],
    )
    distances = load_first_available(
        args.logdir,
        [
            "Metrics/initial_distance_to_goal",
            "Info / Metrics/initial_distance_to_goal",
        ],
    )

    common_steps = sorted(set(returns.keys()) & set(distances.keys()))
    if not common_steps:
        raise SystemExit("No overlapping steps between return and distance scalars.")

    data = defaultdict(list)
    for step in common_steps:
        dist = distances[step]
        ret = returns[step]
        for name, (low, high) in BINS.items():
            if low <= dist < high:
                data[name].append(ret)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (name, _) in zip(axes, BINS.items()):
        values = data.get(name, [])
        ax.hist(values, bins=30, alpha=0.7)
        ax.set_title(f"{name} (n={len(values)})")
        ax.set_xlabel("Episode Return")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Count")
    fig.suptitle("Episode Return by Initial Goal Distance", y=1.02)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
