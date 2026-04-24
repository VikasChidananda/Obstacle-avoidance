# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch


def _resolve_batch_size_and_device(
    obs: torch.Tensor | None, actions: torch.Tensor | None
) -> tuple[int, torch.device | None]:
    """Infer batch size and device from available tensors."""
    batch_size = 0
    device = None
    if obs is not None:
        batch_size = obs.shape[0]
        device = obs.device
    if actions is not None:
        batch_size = actions.shape[0]
        device = actions.device
    return batch_size, device


def _cat_rotations(
    obs: torch.Tensor | None,
    actions: torch.Tensor | None,
    thetas: list[torch.Tensor],
) -> tuple[torch.Tensor | list, torch.Tensor | list]:
    """Concatenate original batch with one rotated copy per theta in thetas."""
    obs_aug: torch.Tensor | list = []
    actions_aug: torch.Tensor | list = []

    if obs is not None:
        obs_parts = [obs]
        for theta in thetas:
            obs_parts.append(yaw_rotate_observation(obs, theta))
        obs_aug = torch.cat(obs_parts, dim=0)

    if actions is not None:
        action_parts = [actions]
        for theta in thetas:
            action_parts.append(yaw_rotate_action(actions, theta))
        actions_aug = torch.cat(action_parts, dim=0)

    return obs_aug, actions_aug


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

    batch_size, device = _resolve_batch_size_and_device(obs, actions)

    if batch_size == 0 or device is None:
        return [], []

    theta = torch.full((batch_size, 1), 0.5 * math.pi, device=device)
    return _cat_rotations(obs, actions, [theta])


def yaw_equivariant_augmentation_c4(
    env,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str | None = None,
) -> tuple[torch.Tensor | list, torch.Tensor | list]:
    """Return C4 augmentation: original + yaw {90, 180, 270} deg (4x batch)."""
    del env, obs_type

    batch_size, device = _resolve_batch_size_and_device(obs, actions)

    if batch_size == 0 or device is None:
        return [], []

    return _yaw_equivariant_augmentation_discretized(obs, actions, batch_size, device, 4)


def yaw_equivariant_augmentation_discretized(
    env,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str | None = None,
) -> tuple[torch.Tensor | list, torch.Tensor | list]:
    """Return evenly spaced yaw augmentation using env.cfg.yaw_discretisations."""
    del obs_type

    batch_size, device = _resolve_batch_size_and_device(obs, actions)

    if batch_size == 0 or device is None:
        return [], []

    num_discretisations = int(getattr(getattr(env, "cfg", None), "yaw_discretisations", 4))
    return _yaw_equivariant_augmentation_discretized(obs, actions, batch_size, device, num_discretisations)


def _yaw_equivariant_augmentation_discretized(
    obs: torch.Tensor | None,
    actions: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    num_discretisations: int,
) -> tuple[torch.Tensor | list, torch.Tensor | list]:
    if num_discretisations < 1:
        raise ValueError("yaw_discretisations must be at least 1")

    if num_discretisations == 1:
        return obs if obs is not None else [], actions if actions is not None else []

    thetas = [
        torch.full((batch_size, 1), 2.0 * math.pi * idx / num_discretisations, device=device)
        for idx in range(1, num_discretisations)
    ]
    return _cat_rotations(obs, actions, thetas)


def yaw_equivariant_augmentation_continuous(
    env,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str | None = None,
) -> tuple[torch.Tensor | list, torch.Tensor | list]:
    """Return original + one random SO(2) yaw rotation per sample (2x batch)."""
    del env, obs_type

    batch_size, device = _resolve_batch_size_and_device(obs, actions)

    if batch_size == 0 or device is None:
        return [], []

    theta = 2.0 * math.pi * torch.rand((batch_size, 1), device=device)
    return _cat_rotations(obs, actions, [theta])


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

# Explicit aliases for augmentation choice in config files.
mirror_augmentation_c4 = yaw_equivariant_augmentation_c4
mirror_augmentation_continuous = yaw_equivariant_augmentation_continuous
