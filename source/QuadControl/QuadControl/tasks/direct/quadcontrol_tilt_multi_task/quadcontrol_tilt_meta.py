# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms
from isaaclab.envs.ui import BaseEnvWindow

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.markers import RED_ARROW_X_MARKER_CFG

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass



from .quadcontrol_env_tilt_cfg import QuadcontrolEnvCfg
# from ..robots.quadcopter import 

from ..quadcontrol_multi_task.quadcontrol_meta_traj import QuadcontrolEnv
from ..trajectory_generator import TrajectoryGenerator


class QuadcontrolTiltEnv(QuadcontrolEnv):
    """The Quadcopter environment for RL training.
    This class is only for hovering with the original implementation. 
    """
    cfg: QuadcontrolEnvCfg

    def __init__(self, cfg: QuadcontrolEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


        self._episode_actions = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "tilt_angle",
                "thrust_z",
                "thrust_y",
                "moment_x",
                "moment_y",
                "moment_z",
            ]
        }


    def _pre_physics_step(self, actions: torch.Tensor):
        # if self.verbose: 
            # print(f"actions: {actions.clone()}")
        self._actions = actions.clone().clamp(-1.0, 1.0)
        # if self.verbose: 
            # print(self._actions, self._actions.shape, self._actions.dtype)

        # map the tilt action to actual angle in radians
        self.tilt_angle = self._actions[:, -1] * self.cfg.angle_scale  # scale   
        # R = self.rotation_matrix_from_tilt(tilt_angle)  # shape (num_envs, 3)


        # self._thrust[:, 0, :] = tilted_thrust_vector   # set the thrust direction
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        
        self._thrust[:, 0, 1] = -self._thrust[:, 0, 2] * torch.sin(self.tilt_angle)# apply rotation to thrust vector
        self._thrust[:, 0, 2] = self._thrust[:, 0, 2] * torch.cos(self.tilt_angle)  # no thrust in x direction for now
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:-1]
        # multiply moment by R
        # self._moment[:, 0, :] = torch.einsum('bij,bj->bi', R, self._moment[:, 0, :])
        # self._moment[:, 0, :] = torch.bmm(R, self._moment[:, 0, :].unsqueeze(-1)).squeeze(-1)

        actions = erk{
            "thrust_z": self._thrust[:, 0, 2],
            "thrust_y": self._thrust[:, 0, 1],
            "moment_x": self._moment[:, 0, 0],
            "moment_y": self._moment[:, 0, 1],
            "moment_z": self._moment[:, 0, 2],
            "tilt_angle": self.tilt_angle,
        }

        for key in self._episode_actions:
            self._episode_actions[key] += actions[key]

        self._episode_errors["control_effort"] += torch.sum(self._actions**2, dim=1)
        
        self.post_physics_step()

erk



    # def _reset_idx(self, env_ids: torch.Tensor | None):
    #     if env_ids is None or len(env_ids) == self.num_envs:
    #         env_ids = self._robot._ALL_INDICES

    #     # Logging
    #     final_distance_to_goal = torch.linalg.norm(
    #         self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
    #     ).mean()
    #     # vel_error = torch.linalg.norm(
    #     #     self._robot.data.root_lin_vel_b[env_ids] - self._desired_vel_w[env_ids], dim=1
    #     # ).mean()

    #     extras = dict()
    #     for key in self._episode_sums.keys():
    #         episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
    #         extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
    #         self._episode_sums[key][env_ids] = 0.0
    #     self.extras["log"] = dict()
    #     self.extras["log"].update(extras)
    #     extras = dict()
    #     extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
    #     extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
    #     extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
    #     self.extras["log"].update(extras)
erk
    #     self._robot.reset(env_ids)
    #     super()._reset_idx(env_ids)

    #     if len(env_ids) == self.num_envs:
    #         # Spread out the resets to avoid spikes in training when many environments reset at a similar time
    #         self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

    #     self.time[env_ids] = 0.0
    #     self._actions[env_ids] = 0.0
    #     # Sample new commands
    #     self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
    #     self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
    #     self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desirederk_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

    #     # 
    #     # --- Desired tilt (orientation) ---
    #     # tilt_roll = torch.zeros(len(env_ids), device=self.device).uniform_(0.0, 0.0)   # radians
    #     # tilt_pitch = torch.zeros(len(env_ids), device=self.device)
    #     # yaw = torch.zeros(len(env_ids), device=self.device)  # keep yaw = 0 for now

    #     def euler_to_quat(roll, pitch, yaw):
    #         cr, sr = torch.cos(roll/2), torch.sin(roll/2)
    #         cp, sp = torch.cos(pitch/2), torch.sin(pitch/2)
    #         cy, sy = torch.cos(yaw/2), torch.sin(yaw/2)
    #         w = cr*cp*cy + sr*sp*sy
    #         x = sr*cp*cy - cr*sp*sy
    #         y = cr*sp*cy + sr*cp*sy
    #         z = cr*cp*sy - sr*sp*cy
    #         return torch.stack([w, x, y, z], dim=-1)

    #     # self._desired_quat_w[env_ids] = euler_to_quat(tilt_roll, tilt_pitch, yaw)
    #     # Reset robot state
    #     joint_pos = self._robot.data.default_joint_pos[env_ids]
    #     joint_vel = self._robot.data.default_joint_vel[env_ids]
    #     default_root_state = self._robot.data.default_root_state[env_ids]
    #     default_root_state[:, :3] += self._terrain.env_origins[env_ids]

    #     # default_root_state[:, 3:7] = self._desired_quat_w[env_ids]

    #     self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    #     self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    #     self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)



    def rotation_matrix_from_tilt(self, angle):
        # Tilt around X axis
        c, s = torch.cos(angle), torch.sin(angle)
        return torch.stack([
            torch.stack([torch.ones_like(angle), torch.zeros_like(angle), torch.zeros_like(angle)], dim=-1),
            torch.stack([torch.zeros_like(angle), c, -s], dim=-1),
            torch.stack([torch.zeros_like(angle), s, c], dim=-1)
        ], dim=-2)

    def get_tilted_thrust(self, tilt_angle, rotor_direction=torch.tensor([0., 0., 1.])):
        # rotor_direction: unit vector in default thrust direction (e.g., [0, 0, 1])
        # tilt_angle: angle in radians
        # Rotate around X or Y axis
        rotor_direction=rotor_direction.to(self.device)
        R = self.rotation_matrix_from_tilt(tilt_angle)
        return R @ rotor_direction
    