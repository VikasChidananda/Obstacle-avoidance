from __future__ import annotations

import math

import torch

from QuadControl.meta_rl.task_sampler import QuadTask, QuadTaskSampler

from .quadcontrol_env import QuadcontrolEnv
from .quadcontrol_meta_env_cfg import QuadcontrolMetaEnvCfg


class QuadcontrolMetaEnv(QuadcontrolEnv):
    """Task-sampled quadrotor env used for meta-learning experiments.

    This class extends the original environment without modifying its behavior.
    Symmetry experiments can continue to use QuadcontrolEnv.
    """

    cfg: QuadcontrolMetaEnvCfg

    def __init__(self, cfg: QuadcontrolMetaEnvCfg, render_mode: str | None = None, verbose: bool = False, **kwargs):
        self.task_sampler = QuadTaskSampler(families=list(cfg.meta_task_families), seed=cfg.task_seed)
        self._forced_task: QuadTask | None = None
        self._active_task: QuadTask = self.task_sampler.sample_task(family=cfg.default_task_family)
        super().__init__(cfg=cfg, render_mode=render_mode, verbose=verbose, **kwargs)
        self._update_desired_pos_from_task()

    def set_meta_task(self, task: QuadTask | None) -> None:
        """Force a task for subsequent resets.

        Passing None reverts to random sampling on reset.
        """
        self._forced_task = task
        if task is not None:
            self._active_task = task
            self._update_desired_pos_from_task()

    def sample_meta_task(self, family: str | None = None) -> QuadTask:
        return self.task_sampler.sample_task(family=family)

    def _pre_physics_step(self, actions: torch.Tensor):
        if hasattr(self, "task_sampler"):
            self._update_desired_pos_from_task()
        super()._pre_physics_step(actions)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if not hasattr(self, "task_sampler"):
            return

        if self._forced_task is not None:
            self._active_task = self._forced_task
        elif self.cfg.sample_new_task_on_reset:
            self._active_task = self.task_sampler.sample_task()

        self._update_desired_pos_from_task(env_ids)

    def _update_desired_pos_from_task(self, env_ids: torch.Tensor | None = None):
        if not hasattr(self, "_terrain"):
            return

        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        time = self.episode_length_buf[env_ids].float() * self.step_dt
        goal_local = self._task_position(self._active_task, time)
        self._desired_pos_w[env_ids] = goal_local + self._terrain.env_origins[env_ids]

    def _task_position(self, task: QuadTask, time: torch.Tensor) -> torch.Tensor:
        params = task.params
        zeros = torch.zeros_like(time)

        if task.family == "hover":
            x = torch.full_like(time, params["x"])
            y = torch.full_like(time, params["y"])
            z = torch.full_like(time, params["height"])

        elif task.family == "straight":
            x = params["x_dir"] * params["speed"] * time
            y = params["y_dir"] * params["speed"] * time
            z = torch.full_like(time, params["height"])

        elif task.family == "circle":
            radius = params["radius"]
            speed = params["speed"]
            z_center = params["z_center"]
            height = params["height"]
            phase = 2.0 * math.pi * speed * time

            x = -radius + radius * torch.cos(phase)
            y = radius * torch.sin(phase)
            z = z_center + height * torch.sin(0.5 * phase)

        elif task.family == "lemniscate":
            a = params["a"]
            speed = params["speed"]
            z_center = params["z_center"]
            phase = 2.0 * math.pi * speed * time

            x = a * torch.sin(phase)
            y = a * torch.sin(phase) * torch.cos(phase)
            z = torch.full_like(time, z_center)

        else:
            x, y, z = zeros, zeros, torch.ones_like(time)

        return torch.stack((x, y, z), dim=1)
