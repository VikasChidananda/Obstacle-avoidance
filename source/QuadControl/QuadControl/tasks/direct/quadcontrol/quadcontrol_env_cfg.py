# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

# from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from ..robots.quadcopter import CRAZYFLIE_CFG


class QuadcontrolEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcontrolEnvCfg, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcontrolEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # - spaces definition
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot(s)
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    thrust_to_weight = 1.9
    moment_scale = 0.01
    yaw_discretisations = 4

    # goal sampling in the local environment frame
    goal_sample_x_range = (0.0, 4.0)
    goal_sample_y_range = (0.0, 4.0)
    goal_sample_z_range = (0.5, 2.0)

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    # ang_vel_reward_scale = -0.5
    distance_to_goal_reward_scale = 15.0

    # debug visualization
    ui_window_class_type = QuadcontrolEnvWindow

    # # task parameters
    # desired_pos_x = [-2.0, 2.0]
    # desired_pos_y = [-2.0, 2.0]
    # desired_pos_z = [0.5, 1.5]


@configclass
class QuadcontrolLineGoalEnvCfg(QuadcontrolEnvCfg):
    """Training config that samples goals on a 1D line segment."""

    goal_sample_x_range = (0.0, 2.0)
    goal_sample_y_range = (0.0, 0.0)
    goal_sample_z_range = (1.0, 1.0)
    eval_log = False


@configclass
class QuadcontrolRandomGoalEnvCfg(QuadcontrolEnvCfg):
    """Evaluation config that samples goals in full 3D."""

    goal_sample_x_range = (0.0, 4.0)
    goal_sample_y_range = (0.0, 4.0)
    goal_sample_z_range = (0.5, 2.0)
    eval_log = True


@configclass
class QuadcontrolQuadrantGoalEnvCfg(QuadcontrolEnvCfg):
    """Training config that samples goals only from the 1st quadrant.

    Goals are restricted to x >= 0, y >= 0 (the positive quadrant in the
    xy-plane) so the agent never sees goals in the other three quadrants
    during training. Generalisation is then evaluated with
    ``QuadcontrolFull3DGoalEnvCfg``.
    """

    goal_sample_x_range = (0.0, 4.0)
    goal_sample_y_range = (0.0, 4.0)
    goal_sample_z_range = (0.5, 2.0)
    eval_log = False


@configclass
class QuadcontrolFull3DGoalEnvCfg(QuadcontrolEnvCfg):
    """Evaluation config that samples goals across the full xy-plane.

    Goals span x in [-4, 4] and y in [-4, 4], i.e. all four quadrants of the
    goal space. Used to measure generalisation of a policy trained only on
    1st-quadrant goals.
    """

    goal_sample_x_range = (-4.0, 4.0)
    goal_sample_y_range = (-4.0, 4.0)
    goal_sample_z_range = (0.5, 2.0)
    eval_log = True
