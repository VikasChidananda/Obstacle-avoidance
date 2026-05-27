# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass

from .quadcontrol_env_cfg import QuadcontrolEnvCfg


@configclass
class QuadcontrolTerrainObstacleEnvCfg(QuadcontrolEnvCfg):
    """Quadrotor obstacle avoidance with procedural terrain obstacles and lidar."""

    observation_space = 17

    # Lidar parameters
    lidar_num_rays = 5
    lidar_max_range = 3.0
    lidar_mesh_prim_path = "/World/ground/terrain"
    lidar_attach_yaw_only = True
    lidar_horizontal_fov_range = (-90.0, 90.0)
    lidar_horizontal_res = 36.0
    lidar_vertical_fov_range = (0.0, 0.0)
    lidar_vis_fraction = 0.1

    # Obstacle avoidance
    obstacle_collision_radius = 0.05
    obstacle_proximity_threshold = 0.5
    obstacle_proximity_penalty_scale = -0.2
    obstacle_collision_penalty = -5.0

    # Spawn and goal sampling
    spawn_height_offset = 1.0
    goal_height_offset = -0.5

    # Termination bounds
    max_height_terminate = 5.5

    # Waypoint traversal
    waypoint_stride = 1
    waypoint_threshold = 0.2
    waypoint_hover_steps = 120
    waypoint_time_steps = 60
    waypoint_end_bonus = 25.0

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(4.0, 4.0),
            num_rows=16,
            num_cols=16,
            border_width=0.2,
            border_height=1.0,
            horizontal_scale=0.1,
            vertical_scale=0.01,
            slope_threshold=0.75,
            sub_terrains={
                "obstacles": HfDiscreteObstaclesTerrainCfg(
                    proportion=1.0,
                    obstacle_width_range=(0.2, 0.6),
                    obstacle_height_range=(0.5, 2.5),
                    num_obstacles=30,
                    platform_width=1.0,
                    flat_patch_sampling={
                        "root_spawn": FlatPatchSamplingCfg(
                            num_patches=32,
                            patch_radius=0.35,
                            max_height_diff=0.03,
                        ),
                        "goal_spawn": FlatPatchSamplingCfg(
                            num_patches=32,
                            patch_radius=0.35,
                            max_height_diff=0.03,
                        ),
                    },
                )
            },
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
