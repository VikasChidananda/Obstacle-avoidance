# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.utils.math import subtract_frame_transforms

from .quadcontrol_env import QuadcontrolEnv
from .quadcontrol_terrain_obstacle_env_cfg import QuadcontrolTerrainObstacleEnvCfg


class QuadcontrolTerrainObstacleEnv(QuadcontrolEnv):
    """Quadrotor obstacle avoidance on procedural terrain with lidar."""

    cfg: QuadcontrolTerrainObstacleEnvCfg

    def __init__(self, cfg: QuadcontrolTerrainObstacleEnvCfg, render_mode: str | None = None, verbose: bool = False, **kwargs):
        self._lidar = None
        self._last_min_range = None
        self._last_collision = None
        self._waypoints_w = None
        self._waypoint_progress = None
        self._waypoint_count = 0
        self._waypoint_y_offsets = None
        self._reached_end = None
        self._lidar_visualizer = None
        self._lidar_vis_mask = None
        super().__init__(cfg, render_mode, verbose, **kwargs)
        if self._lidar is None:
            self._create_lidar_sensor()

        self._last_min_range = torch.full((self.num_envs,), self.cfg.lidar_max_range, device=self.device)
        self._last_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_sums["obstacle"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._waypoint_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._waypoint_y_offsets = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._reached_end = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lidar_vis_mask = self._sample_lidar_vis_mask()

    def _setup_scene(self):
        super()._setup_scene()
        self._create_lidar_sensor()
        self._build_waypoints()

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

        min_range, collided = self._compute_obstacle_metrics()
        self._last_min_range = min_range
        self._last_collision = collided

        proximity = torch.clamp(
            (self.cfg.obstacle_proximity_threshold - min_range) / self.cfg.obstacle_proximity_threshold,
            min=0.0,
            max=1.0,
        )
        proximity_penalty = proximity * self.cfg.obstacle_proximity_penalty_scale * self.step_dt
        collision_penalty = collided.float() * self.cfg.obstacle_collision_penalty * self.step_dt
        obstacle_penalty = proximity_penalty + collision_penalty

        reward = reward + obstacle_penalty
        self._episode_sums["obstacle"] += obstacle_penalty
        self._episode_sums["total"] += obstacle_penalty

        if self._reached_end is not None:
            end_bonus = self._reached_end & (~collided)
            if torch.any(end_bonus):
                reward = reward + end_bonus.float() * self.cfg.waypoint_end_bonus

        self._advance_waypoints()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > self.cfg.max_height_terminate)
        _, collided = self._compute_obstacle_metrics()
        self._last_collision = collided
        died = torch.logical_or(died, collided)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        if len(env_ids) == 0:
            return

        self._waypoint_progress[env_ids] = 0.0
        self._reached_end[env_ids] = False
        if self._waypoints_w is None or self._waypoint_count == 0:
            return

        y_offsets = self._compute_spawn_y_offsets(len(env_ids))
        self._waypoint_y_offsets[env_ids] = y_offsets

        root_pos = self._waypoints_w[0].repeat(len(env_ids), 1)
        root_pos = root_pos.clone()
        root_pos[:, 1] += y_offsets
        root_pos[:, 2] += self.cfg.spawn_height_offset + self.cfg.goal_height_offset

        self._desired_pos_w[env_ids] = self._get_waypoint_positions(env_ids, torch.zeros(len(env_ids), dtype=torch.long, device=self.device))

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] = root_pos

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        init_distance = torch.linalg.norm(self._desired_pos_w[env_ids, :2] - default_root_state[:, :2], dim=1)
        self._episode_init_distance[env_ids] = init_distance

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
        lidar_ranges = self._compute_lidar_ranges()
        min_range, _ = torch.min(lidar_ranges, dim=1)
        collided = min_range < self.cfg.obstacle_collision_radius
        return min_range, collided

    def _build_waypoints(self):
        if self._terrain.terrain_origins is not None:
            origins = self._terrain.terrain_origins
            row_count = origins.shape[0]
            row_ids = torch.arange(0, row_count, self.cfg.waypoint_stride, device=self.device)
            if row_ids[-1] != row_count - 1:
                row_ids = torch.cat([row_ids, torch.tensor([row_count - 1], device=self.device)])
            waypoints = origins[row_ids, 0, :].to(self.device)
            waypoints[:, 1] = 0.0
        else:
            base_origin = self._terrain.env_origins[0]
            generator_cfg = self.cfg.terrain.terrain_generator
            row_count = generator_cfg.num_rows
            row_ids = torch.arange(0, row_count, self.cfg.waypoint_stride, device=self.device)
            if row_ids[-1] != row_count - 1:
                row_ids = torch.cat([row_ids, torch.tensor([row_count - 1], device=self.device)])
            total_length = generator_cfg.size[0] * row_count
            x = base_origin[0] - 0.5 * total_length + (row_ids + 0.5) * generator_cfg.size[0]
            y = torch.full_like(x, base_origin[1])
            z = torch.full_like(x, base_origin[2])
            waypoints = torch.stack([x, y, z], dim=1)

        waypoints = waypoints.clone()
        waypoints[:, 2] += self.cfg.goal_height_offset
        self._waypoints_w = waypoints
        self._waypoint_count = waypoints.shape[0]
        if self._waypoint_progress is None:
            self._waypoint_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _advance_waypoints(self):
        if self._waypoints_w is None or self._waypoint_count == 0:
            return

        can_advance = self.episode_length_buf >= self.cfg.waypoint_hover_steps
        step_increment = 1.0 / max(1, self.cfg.waypoint_time_steps)
        
        new_progress = self._waypoint_progress + can_advance.float() * step_increment
        new_progress = torch.clamp(new_progress, max=self._waypoint_count - 1.0)
        self._waypoint_progress = new_progress

        wp_indices_low = torch.floor(new_progress).to(torch.long)
        wp_indices_high = torch.ceil(new_progress).to(torch.long)
        alpha = new_progress - wp_indices_low
        
        wp_low = self._get_waypoint_positions(self._robot._ALL_INDICES, wp_indices_low)
        wp_high = self._get_waypoint_positions(self._robot._ALL_INDICES, wp_indices_high)
        
        self._desired_pos_w = wp_low + (wp_high - wp_low) * alpha.unsqueeze(1)
        
        # Check if reached end
        reached_end = (new_progress >= self._waypoint_count - 1.0)
        self._reached_end = torch.logical_or(self._reached_end, reached_end)

    def _compute_spawn_y_offsets(self, count: int) -> torch.Tensor:
        generator_cfg = self.cfg.terrain.terrain_generator
        total_width = generator_cfg.size[1] * generator_cfg.num_cols
        y_min = -0.5 * total_width + 0.5 * generator_cfg.size[1]
        y_max = 0.5 * total_width - 0.5 * generator_cfg.size[1]
        if count <= 0:
            return torch.zeros(0, device=self.device)
        return torch.empty(count, device=self.device).uniform_(y_min, y_max)

    def _get_waypoint_positions(self, env_ids: torch.Tensor, waypoint_indices: torch.Tensor) -> torch.Tensor:
        base_wp = self._waypoints_w[waypoint_indices]
        base_wp = base_wp.clone()
        base_wp[:, 1] += self._waypoint_y_offsets[env_ids]
        return base_wp

    def _sample_lidar_vis_mask(self) -> torch.Tensor:
        if self.cfg.lidar_vis_fraction <= 0.0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.lidar_vis_fraction >= 1.0:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.rand(self.num_envs, device=self.device) < self.cfg.lidar_vis_fraction

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if self._lidar_visualizer is None:
                marker_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/LidarHits")
                self._lidar_visualizer = VisualizationMarkers(marker_cfg)
            self._lidar_visualizer.set_visibility(True)
        elif self._lidar_visualizer is not None:
            self._lidar_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if self._lidar_visualizer is None or self._lidar is None:
            return
        if self._lidar_vis_mask is None:
            self._lidar_vis_mask = self._sample_lidar_vis_mask()

        hits_w = self._lidar.data.ray_hits_w[self._lidar_vis_mask]
        if hits_w.numel() == 0:
            return
        viz_points = hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        if viz_points.numel() == 0:
            return
        self._lidar_visualizer.visualize(viz_points)
