from __future__ import annotations

from isaaclab.utils import configclass

from .quadcontrol_env_cfg import QuadcontrolEnvCfg


@configclass
class QuadcontrolMetaEnvCfg(QuadcontrolEnvCfg):
    """Meta-RL specific config for task-sampled quadrotor tracking."""

    # Ensure state_space is defined for skrl model initialization (STATES input).
    state_space = 12
    meta_task_families: tuple[str, ...] = ("hover", "straight", "circle", "lemniscate")
    default_task_family: str = "hover"
    sample_new_task_on_reset: bool = True
    task_seed: int | None = None
