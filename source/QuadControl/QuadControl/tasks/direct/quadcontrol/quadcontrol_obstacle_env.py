# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.warp import convert_to_warp_mesh

from .quadcontrol_env import QuadcontrolEnv
from .quadcontrol_obstacle_env_cfg import QuadcontrolObstacleCurriculumEnvCfg, QuadcontrolObstacleEnvCfg

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcontrolObstacleEnv(QuadcontrolEnv):
    """Quadrotor navigation environment with lidar and obstacles."""

    cfg: QuadcontrolObstacleEnvCfg

    def __init__(self, cfg: QuadcontrolObstacleEnvCfg, render_mode: str | None = None, verbose: bool = False, **kwargs):
        # Ensure debug visualization can run during base init.
        self._lidar = None
        self._obstacle_visualizer = None
        self._obstacle_positions = None
        self._obstacle_sizes = None
        self._obstacle_active_counts = None
        super().__init__(cfg, render_mode, verbose, **kwargs)
        if self._lidar is None:
            self._create_lidar_sensor()

        self._lidar_mesh_dirty = True
        self._lidar_mesh_signs = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            device=self.device,
        )
        self._lidar_mesh_faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=np.int32,
        )

        self._obstacle_positions = torch.zeros(
            self.num_envs, self.cfg.obstacle_count_max, 2, device=self.device
        )
        self._obstacle_sizes = torch.zeros(
            self.num_envs, self.cfg.obstacle_count_max, 2, device=self.device
        )
        self._obstacle_active_counts = torch.full(
            (self.num_envs,), self.cfg.obstacle_count_min, dtype=torch.long, device=self.device
        )
        self._global_reset_count = 0
        self._corridor_segments = self._compute_corridor_segments()
        self._corridor_obstacle_count = self._corridor_segments * 2
        self.cfg.obstacle_height = self.cfg.corridor_wall_height
        if self._corridor_obstacle_count > self.cfg.obstacle_count_max:
            raise ValueError(
                "Corridor needs more obstacles than obstacle_count_max. "
                f"needed={self._corridor_obstacle_count}, obstacle_count_max={self.cfg.obstacle_count_max}"
            )
        self._last_obstacle_distance = torch.full(
            (self.num_envs,), self.cfg.lidar_max_range, device=self.device
        )
        self._last_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._episode_sums["obstacle"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _setup_scene(self):
        super()._setup_scene()
        self._create_lidar_sensor()

    def _create_lidar_sensor(self):
        if self._lidar is not None:
            return

        lidar_pattern = LidarPatternCfg(
            channels=1,
            vertical_fov_range=self.cfg.lidar_vertical_fov_range,
            horizontal_fov_range=self.cfg.lidar_horizontal_fov_range,
            horizontal_res=self.cfg.lidar_horizontal_res,
        )
        expected_rays = lidar_pattern.func(lidar_pattern, self.device)[1].shape[0]
        if expected_rays != self.cfg.lidar_num_rays:
            raise ValueError(
                "Lidar pattern does not match lidar_num_rays. "
                f"pattern_rays={expected_rays}, lidar_num_rays={self.cfg.lidar_num_rays}"
            )

        lidar_cfg = RayCasterCfg(
            prim_path=self.cfg.robot.prim_path,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            attach_yaw_only=self.cfg.lidar_attach_yaw_only,
            pattern_cfg=lidar_pattern,
            max_distance=self.cfg.lidar_max_range,
            mesh_prim_paths=[self.cfg.lidar_mesh_prim_path],
            drift_range=(0.0, 0.0),
        )
        self._lidar = RayCaster(lidar_cfg)
        if not hasattr(self._lidar, "drift"):
            self._lidar.drift = torch.zeros(self.num_envs, 3, device=self.device)
        self.scene.sensors["lidar"] = self._lidar

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        lidar_ranges = self._compute_lidar_ranges()

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                lidar_ranges,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()

        min_distance, collided = self._compute_obstacle_metrics()
        self._last_obstacle_distance = min_distance
        self._last_collision = collided

        proximity = torch.clamp(
            (self.cfg.obstacle_proximity_threshold - min_distance) / self.cfg.obstacle_proximity_threshold,
            min=0.0,
            max=1.0,
        )
        proximity_penalty = proximity * self.cfg.obstacle_proximity_penalty_scale * self.step_dt
        collision_penalty = collided.float() * self.cfg.obstacle_collision_penalty * self.step_dt
        obstacle_penalty = proximity_penalty + collision_penalty

        reward = reward + obstacle_penalty
        self._episode_sums["obstacle"] += obstacle_penalty
        self._episode_sums["total"] += obstacle_penalty

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died, time_out = super()._get_dones()
        _, collided = self._compute_obstacle_metrics()
        self._last_collision = collided
        died = torch.logical_or(died, collided)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == 0:
            return

        env_origins = self._terrain.env_origins[env_ids]
        self._desired_pos_w[env_ids, 0] = env_origins[:, 0] + self.cfg.corridor_length
        self._desired_pos_w[env_ids, 1] = env_origins[:, 1]
        self._desired_pos_w[env_ids, 2] = self.cfg.corridor_goal_height

        self._global_reset_count += len(env_ids)
        self._sample_obstacles(env_ids)
        self._update_obstacle_markers()

    def _sample_obstacles(self, env_ids: torch.Tensor):
        if env_ids is None or len(env_ids) == 0:
            return

        env_origins = self._terrain.env_origins[env_ids, :2]
        base_positions, base_sizes = self._build_corridor_layout()

        obstacle_positions = base_positions[None, :, :] + env_origins[:, None, :]
        obstacle_sizes = base_sizes[None, :, :].expand(len(env_ids), -1, -1)

        self._obstacle_positions[env_ids] = 0.0
        self._obstacle_sizes[env_ids] = 0.0
        self._obstacle_positions[env_ids, : self._corridor_obstacle_count] = obstacle_positions
        self._obstacle_sizes[env_ids, : self._corridor_obstacle_count] = obstacle_sizes
        self._obstacle_active_counts[env_ids] = self._corridor_obstacle_count
        self._lidar_mesh_dirty = True

    def _curriculum_progress(self) -> float:
        if not self.cfg.obstacle_curriculum:
            return 1.0
        return min(1.0, self._global_reset_count / float(self.cfg.obstacle_curriculum_steps))

    def _compute_active_obstacle_count(self, progress: float) -> int:
        min_count = self.cfg.obstacle_count_min
        max_count = self.cfg.obstacle_count_max
        return int(round(min_count + (max_count - min_count) * progress))

    def _compute_corridor_segments(self) -> int:
        return max(1, int(math.ceil(self.cfg.corridor_length / self.cfg.corridor_segment_length)))

    def _build_corridor_layout(self) -> tuple[torch.Tensor, torch.Tensor]:
        segments = self._corridor_segments
        seg_len = self.cfg.corridor_segment_length
        x_positions = torch.linspace(
            seg_len * 0.5,
            self.cfg.corridor_length - seg_len * 0.5,
            segments,
            device=self.device,
        )
        y_offset = 0.5 * self.cfg.corridor_width + 0.5 * self.cfg.corridor_wall_thickness

        x = x_positions.repeat_interleave(2)
        y = torch.tile(
            torch.tensor([y_offset, -y_offset], device=self.device),
            (segments,),
        )
        positions = torch.stack([x, y], dim=-1)

        size_x = seg_len * 0.5
        size_y = self.cfg.corridor_wall_thickness * 0.5
        sizes = torch.zeros(positions.shape[0], 2, device=self.device)
        sizes[:, 0] = size_x
        sizes[:, 1] = size_y
        return positions, sizes

    def _compute_lidar_ranges(self) -> torch.Tensor:
        if self._lidar is None:
            self._create_lidar_sensor()

        if self._lidar is None or not self._lidar.is_initialized:
            return torch.full(
                (self.num_envs, self.cfg.lidar_num_rays),
                self.cfg.lidar_max_range,
                device=self.device,
            )

        lidar_data = self._lidar.data
        if self._lidar_mesh_dirty and self._lidar.is_initialized:
            self._update_lidar_mesh()
            self._lidar.update(0.0, force_recompute=True)
            lidar_data = self._lidar.data

        hits_w = lidar_data.ray_hits_w
        origins_w = lidar_data.pos_w[:, None, :]
        ranges = torch.linalg.norm(hits_w - origins_w, dim=-1)
        ranges = torch.where(
            torch.isfinite(ranges),
            ranges,
            torch.full_like(ranges, self.cfg.lidar_max_range),
        )
        return torch.clamp(ranges, max=self.cfg.lidar_max_range)

    def _compute_obstacle_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_xy = self._robot.data.root_pos_w[:, :2]
        root_z = self._robot.data.root_pos_w[:, 2]

        obs_positions = self._obstacle_positions
        obs_sizes = self._obstacle_sizes

        idx = torch.arange(self.cfg.obstacle_count_max, device=self.device).unsqueeze(0)
        active_mask = idx < self._obstacle_active_counts.unsqueeze(1)

        min_xy = obs_positions - obs_sizes
        max_xy = obs_positions + obs_sizes

        dx = torch.maximum(min_xy[..., 0] - root_xy[:, None, 0], root_xy[:, None, 0] - max_xy[..., 0])
        dy = torch.maximum(min_xy[..., 1] - root_xy[:, None, 1], root_xy[:, None, 1] - max_xy[..., 1])
        dx = torch.clamp(dx, min=0.0)
        dy = torch.clamp(dy, min=0.0)

        dist = torch.sqrt(dx**2 + dy**2)
        dist = torch.where(active_mask, dist, torch.full_like(dist, self.cfg.lidar_max_range))
        min_dist, _ = torch.min(dist, dim=1)

        in_height = root_z <= self.cfg.obstacle_height
        collided = (min_dist < self.cfg.obstacle_collision_radius) & in_height
        return min_dist, collided

    def _update_lidar_mesh(self):
        if self._lidar is None or not self._lidar.meshes:
            return

        points, indices = self._build_obstacle_mesh()
        if points is None:
            return

        mesh_path = self._lidar.cfg.mesh_prim_paths[0]
        self._lidar.meshes[mesh_path] = convert_to_warp_mesh(points, indices, device=self.device)
        self._lidar_mesh_dirty = False

    def _build_obstacle_mesh(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        idx = torch.arange(self.cfg.obstacle_count_max, device=self.device).unsqueeze(0)
        active_mask = idx < self._obstacle_active_counts.unsqueeze(1)
        if not active_mask.any():
            return None, None

        positions = self._obstacle_positions[active_mask]
        sizes = self._obstacle_sizes[active_mask]
        num_obs = positions.shape[0]

        half_height = self.cfg.obstacle_height * 0.5
        centers = torch.cat(
            [positions, torch.full((num_obs, 1), half_height, device=self.device)],
            dim=1,
        )
        half_extents = torch.cat(
            [sizes, torch.full((num_obs, 1), half_height, device=self.device)],
            dim=1,
        )

        verts = centers[:, None, :] + half_extents[:, None, :] * self._lidar_mesh_signs
        verts = verts.reshape(-1, 3).cpu().numpy()

        face_offsets = (np.arange(num_obs, dtype=np.int32) * 8)[:, None, None]
        faces = self._lidar_mesh_faces[None, :, :] + face_offsets
        faces = faces.reshape(-1, 3)
        return verts, faces

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if self._obstacle_visualizer is None:
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (
                    self.cfg.corridor_segment_length,
                    self.cfg.corridor_wall_thickness,
                    self.cfg.corridor_wall_height,
                )
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = self.cfg.obstacle_color
                marker_cfg.prim_path = "/Visuals/Obstacles"
                self._obstacle_visualizer = VisualizationMarkers(marker_cfg)
            self._obstacle_visualizer.set_visibility(True)
            self._update_obstacle_markers()
        elif self._obstacle_visualizer is not None:
            self._obstacle_visualizer.set_visibility(False)

    def _update_obstacle_markers(self):
        if self._obstacle_visualizer is None:
            return
        if self._obstacle_positions is None or self._obstacle_active_counts is None:
            return

        positions = self._obstacle_positions.clone()
        idx = torch.arange(self.cfg.obstacle_count_max, device=self.device).unsqueeze(0)
        active_mask = idx < self._obstacle_active_counts.unsqueeze(1)

        z = torch.full_like(positions[..., :1], self.cfg.obstacle_height * 0.5)
        positions = torch.cat([positions, z], dim=-1)

        hidden = torch.tensor([0.0, 0.0, -100.0], device=self.device)
        positions = torch.where(active_mask[..., None], positions, hidden)

        self._obstacle_visualizer.visualize(positions.view(-1, 3))


class QuadcontrolObstacleCurriculumEnv(QuadcontrolObstacleEnv):
    """Obstacle env with curriculum-driven difficulty increases."""

    cfg: QuadcontrolObstacleCurriculumEnvCfg
