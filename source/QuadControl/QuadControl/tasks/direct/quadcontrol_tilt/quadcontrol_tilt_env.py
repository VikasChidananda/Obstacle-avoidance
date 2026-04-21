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

class QuadcontrolEnv(DirectRLEnv):
    """The Quadcopter environment for RL training.
    This class is only for hovering with the original implementation. 
    """
    cfg: QuadcontrolEnvCfg

    def __init__(self, cfg: QuadcontrolEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.verbose = False
        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        
        if self.verbose: 
            print(f"Actions shape:, {self._actions.shape} \n  num_envs: {self.num_envs}, action_space: {self.single_action_space}")
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
      
        # set vel_x = 0.1 - need all crazy flies to hover and follow forward in x direction
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "orientation_error",
                # "action_penalty"
            ]
        }
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

        self._episode_errors = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "tracking_cost",
                "control_effort",
            ]
        }


        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        
        self.previous_actions = torch.zeros_like(self._actions, device=self.device)
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.verbose: 
            print(f"actions: {actions.clone()}")
        self._actions = actions.clone().clamp(-1.0, 1.0)
        if self.verbose: 
            print(self._actions, self._actions.shape, self._actions.dtype)

        # map the tilt action to actual angle in radians
        tilt_angle = self._actions[:, -1] * self.cfg.angle_scale  # scale   
        R = self.rotation_matrix_from_tilt(tilt_angle)  # shape (num_envs, 3)


        # self._thrust[:, 0, :] = tilted_thrust_vector   # set the thrust direction
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._thrust[:, 0, 1] = -self._thrust[:, 0, 2] * torch.sin(tilt_angle)# apply rotation to thrust vector
        self._thrust[:, 0, 2] = self._thrust[:, 0, 2] * torch.cos(tilt_angle)  # no thrust in x direction for now
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:-1]
        # multiply moment by R
        # self._moment[:, 0, :] = torch.einsum('bij,bj->bi', R, self._moment[:, 0, :])
        self._moment[:, 0, :] = torch.bmm(R, self._moment[:, 0, :].unsqueeze(-1)).squeeze(-1)

        forces = {
            "tilt_angle": tilt_angle,
            "thrust_z": self._thrust[:, 0, 2],
            "thrust_y": self._thrust[:, 0, 1],
            "moment_x": self._moment[:, 0, 0],
            "moment_y": self._moment[:, 0, 1],
            "moment_z": self._moment[:, 0, 2],
        }

        for key in forces:
            self._episode_actions[key] += forces[key]
        
        self._episode_errors["control_effort"] += torch.sum(self._actions**2, dim=1)


    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        if self.verbose: 
            print(f"desired pos in body frame: {desired_pos_b.shape} \t action: {self._actions.shape}")

        # add tilt angle and quaternions to observations
        # desired_quat_b = quat_multiply(quat_conjugate(self._robot.data.root_state_w[:, 3:7]), self._desired_quat_w)
        # obs =
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                # self._actions,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        if self.verbose: 
            print(f"observations limits: [{obs.max().item(), obs.min().item()}]")
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        action_penalty = torch.sum(torch.square(self._actions - self.previous_actions), dim=1)

        orientation_error = 1 - torch.sum(self._robot.data.root_state_w[:, 3:7] * self._desired_quat_w, dim=1)**2
        
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            # "action_penalty": -0.01 * action_penalty * self.step_dt
            "orientation_error": orientation_error * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # reward = torch.clip(reward, -1.0, 1.0)
        # reward = normalise_reward(reward)

        self._episode_errors["tracking_cost"] += distance_to_goal

        # if self.verbose: 
        #     print(f"rewards: {reward}")
        # # Logging

        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        # vel_error = torch.linalg.norm(
        #     self._robot.data.root_lin_vel_b[env_ids] - self._desired_vel_w[env_ids], dim=1
        # ).mean()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()

        for key in self._episode_actions.keys():
            episodic_action_avg = torch.mean(self._episode_actions[key][env_ids])
            extras["Episode_Action/" + key] = episodic_action_avg / self.max_episode_length_s
            self._episode_actions[key][env_ids] = 0.0

        for key in self._episode_errors.keys():
            episodic_error_avg = torch.mean(self._episode_errors[key][env_ids])
            extras["Episode_Error/" + key] = episodic_error_avg / self.max_episode_length_s
            self._episode_errors[key][env_ids] = 0.0

        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids])
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids])
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal

        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # --- Desired tilt (orientation) ---
        tilt_roll = torch.zeros(len(env_ids), device=self.device).uniform_(0.0, 0.0)   # radians
        tilt_pitch = torch.zeros(len(env_ids), device=self.device)
        yaw = torch.zeros(len(env_ids), device=self.device)  # keep yaw = 0 for now

        def euler_to_quat(roll, pitch, yaw):
            cr, sr = torch.cos(roll/2), torch.sin(roll/2)
            cp, sp = torch.cos(pitch/2), torch.sin(pitch/2)
            cy, sy = torch.cos(yaw/2), torch.sin(yaw/2)
            w = cr*cp*cy + sr*sp*sy
            x = sr*cp*cy - cr*sp*sy
            y = cr*sp*cy + sr*cp*sy
            z = cr*cp*sy - sr*sp*cy
            return torch.stack([w, x, y, z], dim=-1)

        self._desired_quat_w[env_ids] = euler_to_quat(tilt_roll, tilt_pitch, yaw)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 3:7] = self._desired_quat_w[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)

            # velocity visualizer
            # marker_cfg
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     # create markers if necessary for the first tome
    #     if debug_vis:
    #         if not hasattr(self, "goal_pos_visualizer"):
    #             # marker_cfg = VisualizationMarkersCfg(
    #             # prim_path="/Visuals/Command/goal_position",
    #             # markers={
    #             #     "arrow": sim_utils.MarkerCfg(
    #             #         usd_path=f"{ISAAC_NUCLEUS_DIR}/"
    #             # marker_type="frame",  # or "arrow"
    #             # scale=(0.1, 0.1, 0.1),
    #             # color=(1.0, 0.0, 0.0)  # RGB for red
    #             # )

    #             # marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
    #             marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
    #             # -- goal pose
    #             marker_cfg.prim_path = "/Visuals/Command/goal_position"
    #             self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
    #         # set their visibility to true
    #         self.goal_pos_visualizer.set_visibility(True)

    #         # velocity visualizer
    #         # marker_cfg
    #     else:
    #         if hasattr(self, "goal_pos_visualizer"):
    #             self.goal_pos_visualizer.set_visibility(False)



    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w, self._desired_quat_w)

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
        rotor_direction = rotor_direction.to(self.device)
        R = self.rotation_matrix_from_tilt(tilt_angle)
        return R @ rotor_direction
    