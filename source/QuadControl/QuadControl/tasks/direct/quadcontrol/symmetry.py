# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch


def yaw_equivariant_augmentation(
    env,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str | None = None,
) -> tuple[torch.Tensor | list, torch.Tensor | list]:
    """Return original + yaw-rotated samples for symmetry-aware PPO updates.

    We use a deterministic 90 deg yaw element (C4 subgroup of SO(2)) so the same
    transform can be reused reliably by both data augmentation and mirror loss.
    """
    del env, obs_type

    obs_aug: torch.Tensor | list = []
    actions_aug: torch.Tensor | list = []

    batch_size = 0
    device = None
    if obs is not None:
        batch_size = obs.shape[0]
        device = obs.device
    if actions is not None:
        batch_size = actions.shape[0]
        device = actions.device

    if batch_size == 0 or device is None:
        return obs_aug, actions_aug

    theta = torch.full((batch_size, 1), 0.5 * math.pi, device=device)

    if obs is not None:
        obs_aug = torch.cat([obs, yaw_rotate_observation(obs, theta)], dim=0)

    if actions is not None:
        actions_aug = torch.cat([actions, yaw_rotate_action(actions, theta)], dim=0)

    return obs_aug, actions_aug


def yaw_rotate_observation(obs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate policy observations around z axis by yaw angle theta.

    Observation layout in this task:
    [lin_vel_b(3), ang_vel_b(3), projected_gravity_b(3), desired_pos_b(3)].
    """
    obs_rot = obs.clone()
    cos_t = torch.cos(theta[:, 0])
    sin_t = torch.sin(theta[:, 0])

    for start in (0, 3, 6, 9):
        x = obs[:, start]
        y = obs[:, start + 1]
        obs_rot[:, start] = cos_t * x - sin_t * y
        obs_rot[:, start + 1] = sin_t * x + cos_t * y

    return obs_rot


def yaw_rotate_action(actions: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate action moments under yaw.

    Action layout: [thrust, roll_moment, pitch_moment, yaw_moment].
    The in-plane moment vector [roll, pitch] rotates with yaw.
    """
    actions_rot = actions.clone()
    cos_t = torch.cos(theta[:, 0])
    sin_t = torch.sin(theta[:, 0])

    roll = actions[:, 1]
    pitch = actions[:, 2]
    actions_rot[:, 1] = cos_t * roll - sin_t * pitch
    actions_rot[:, 2] = sin_t * roll + cos_t * pitch
    return actions_rot


# Backward-compatible aliases for old config names.
mirror_augmentation = yaw_equivariant_augmentation
mirror_observation = yaw_rotate_observation
mirror_action = yaw_rotate_action
