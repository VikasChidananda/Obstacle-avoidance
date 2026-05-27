# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from .quadcontrol_env_cfg import QuadcontrolEnvCfg


@configclass
class QuadcontrolObstacleEnvCfg(QuadcontrolEnvCfg):
    """Quadrotor navigation with obstacles and a 5-ray lidar."""

    observation_space = 17

    # Lidar parameters
    lidar_num_rays = 5
    lidar_max_range = 3.0
    lidar_mesh_prim_path = "/World/ground"
    lidar_attach_yaw_only = True
    lidar_horizontal_fov_range = (-90.0, 90.0)
    lidar_horizontal_res = 36.0
    lidar_vertical_fov_range = (0.0, 0.0)

    # Corridor parameters
    corridor_length = 20.0
    corridor_width = 2.0
    corridor_wall_thickness = 0.3
    corridor_wall_height = 2.0
    corridor_segment_length = 1.0
    corridor_goal_height = 1.0
    obstacle_color = (0.55, 0.27, 0.07)

    # Obstacle layout
    obstacle_count_min = 40
    obstacle_count_max = 40
    obstacle_spawn_radius = 3.5
    obstacle_size_min = 0.12
    obstacle_size_max = 0.35
    obstacle_height = 0.7
    obstacle_min_start_clearance = 0.6
    obstacle_min_goal_clearance = 0.6

    # Obstacle avoidance
    obstacle_collision_radius = 0.12
    obstacle_proximity_threshold = 0.5
    obstacle_proximity_penalty_scale = -0.2
    obstacle_collision_penalty = -5.0

    # Curriculum settings
    obstacle_curriculum = False
    obstacle_curriculum_steps = 2000


@configclass
class QuadcontrolObstacleCurriculumEnvCfg(QuadcontrolObstacleEnvCfg):
    """Obstacle env with curriculum that increases difficulty over training."""

    obstacle_curriculum = True
