# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .quadcontrol_env_cfg import QuadcontrolEnvCfg


@configclass
class QuadcontrolTreeEnvCfg(QuadcontrolEnvCfg):
    """Quadrotor obstacle avoidance using a custom USD tree environment with lidar."""

    observation_space = 17

    # Lidar parameters
    lidar_num_rays = 5
    lidar_max_range = 3.0
    lidar_mesh_prim_path = "/World/terrain/forest/Objects"  # USD tree assets live under /World/Vegetation
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
    spawn_height_offset = 1.0  # Set default root state to be at 1.0m
    goal_height_offset = 0.0

    # Termination bounds
    max_height_terminate = 5.5

    # Waypoint traversal
    waypoint_stride = 1
    waypoint_threshold = 0.2
    waypoint_hover_steps = 120
    waypoint_time_steps = 60
    waypoint_end_bonus = 25.0
    
    # Environment dimension for waypoint calculation (assuming similar to long terrain)
    env_length = 32.0 
    env_width = 8.0
    num_waypoints = 16

    terrain = TerrainImporterCfg(
        prim_path="/World",
        terrain_type="usd",
        usd_path=os.path.join(os.path.dirname(__file__), "forest.usd"),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
