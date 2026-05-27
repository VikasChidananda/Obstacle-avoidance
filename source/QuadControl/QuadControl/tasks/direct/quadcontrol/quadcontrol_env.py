# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaacsim.core.utils.prims import create_prim

# from isaacsim.core.utils.prims import create_prim
# from isaaclab.sim import create_prim
from torch.utils.tensorboard import SummaryWriter

from .quadcontrol_env_cfg import QuadcontrolEnvCfg

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip



# quadrotor environment configuration comes from this paper:
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8967695
# Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to
# Multiple Quadrotors



class QuadcontrolEnv(DirectRLEnv):
    """The Quadcopter environment for RL training.
    This class is only for hovering with the original implementation.

    """

    cfg: QuadcontrolEnvCfg

    def __init__(self, cfg: QuadcontrolEnvCfg, render_mode: str | None = None, verbose=False, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.verbose = verbose

        if self.verbose:
            print(
                f"Actions shape:, {self._actions.shape} \n  num_envs: {self.num_envs}, action_space:"
                f" {self.single_action_space}"
            )

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.eval_log = getattr(self.cfg, "eval_log", False)
        if self.eval_log:
            self.eval_log_dir = Path(self.cfg.log_dir) / "eval_logs"
            self.eval_log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Evaluation logging enabled. Log dir: {self.eval_log_dir}")

            self._traj_log = []
        self.died = [False]
        self.time_out = [False]
        self.eps = 0

        # self._log = {

        # }
        # set vel_x = 0.1 - need all crazy flies to hover and follow forward in x direction
        # for env_id in range(self.num_envs):
        # self.writer = SummaryWriter(log_dir="./logs")
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "total",
                # "action_penalty"
            ]
        }
        self._episode_init_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.previous_actions = torch.zeros_like(self._actions, device=self.device)
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self._hist_writer = None
        self._hist_step = 0
        # # if hasattr(self.cfg, "log_dir") and self.cfg.log_dir:
        # hist_dir = Path(self.cfg.log_dir) / "env_histograms"
        # hist_dir.mkdir(parents=True, exist_ok=True)
        # self._hist_writer = SummaryWriter(log_dir=str(hist_dir))

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
            print(f"actions: {actions.clone().shape}")
        self._actions = actions.clone().clamp(-1.0, 1.0)
        if self.verbose:
            print(self._actions, self._actions.shape, self._actions.dtype)

        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        self.post_physics_step()

    def post_physics_step(self):
        if not self.eval_log:
            return  # no logging in training

        # collect state for env 0
        print(self.episode_length_buf[0], self.max_episode_length)
        rs = self._robot.data.root_state_w  # (num_envs, 13): pos, quat, lin_vel, ang_vel[web:3]
        pos = rs[0, 0:3]
        quat = rs[0, 3:7]
        lin_vel = rs[0, 7:10]
        ang_vel = rs[0, 10:13]
        ref_pos = self._desired_pos_w[0]

        self._traj_log.append(
            dict(
                t=float(self.episode_length_buf[0] * self.step_dt),
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]),
                qx=float(quat[0]),
                qy=float(quat[1]),
                qz=float(quat[2]),
                qw=float(quat[3]),
                vx=float(lin_vel[0]),
                vy=float(lin_vel[1]),
                vz=float(lin_vel[2]),
                wx=float(ang_vel[0]),
                wy=float(ang_vel[1]),
                wz=float(ang_vel[2]),
                xref=float(ref_pos[0]),
                yref=float(ref_pos[1]),
                zref=float(ref_pos[2]),
            )
        )

        done_max = self.died[0] or self.time_out[0]
        if done_max:
            # self.died = self.episode_length_buf[0]
            print("evaluation episode done, saving trajectory log...")
            # save trajectory log to file
            self.save_eval_trajectory(tag=f"eval_episode_{self.eps}")
            # self. = []  # reset for next episode

    def save_eval_trajectory(self, tag="episode0"):
        import json
        import time

        self.eps += 1
        if not self.eval_log or len(self._traj_log) == 0:
            return
        print("saving eval trajectory log...")
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = self.eval_log_dir / f"{tag}_{ts}.json"
        with path.open("w") as f:
            json.dump(self._traj_log, f)
        self._traj_log = []

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        if self.verbose:
            print(f"desired pos in body frame: {desired_pos_b.shape} \t action: {self._actions.shape}")
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

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            # "action_penalty": -0.01 * action_penalty * self.step_dt
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # reward = torch.clip(reward, -1.0, 1.0)
        # reward = normalise_reward(reward)

        if self.verbose:
            print(f"rewards: {reward}")
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["total"] += reward
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 3.0)
        # print(f"died: {died}, time_out: {time_out}")
        self.died, self.time_out = died, time_out
        return died, time_out

    def _spawn_traj_point(self, env_id, pos):
        prim_path = f"/World/QuadcopterTrajPts/env_{env_id}_pt_{len(self.traj_points[env_id])}"
        create_prim(
            prim_path=prim_path,
            prim_type="Sphere",
            position=pos,
            attributes={"radius": 0.01},
        )

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
        per_env_total_return = self._episode_sums["total"][env_ids].detach().cpu()
        per_env_init_distance = self._episode_init_distance[env_ids].detach().cpu()
        total_return_mean = torch.mean(self._episode_sums["total"][env_ids])
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Return/total"] = total_return_mean
        extras["Metrics/initial_distance_to_goal"] = torch.mean(self._episode_init_distance[env_ids])
        self.extras["log"] = dict()
        # self.extras["log"].update(extras)

        # extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids])
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids])
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal
        if self.verbose:
            print(f"logging extras on reset: {extras}")
        self.extras["log"].update(extras)

        if self._hist_writer is not None:
            self._hist_writer.add_histogram(
                "Episode_Return/total_per_env",
                per_env_total_return,
                self._hist_step,
            )
            self._hist_writer.add_histogram(
                "Metrics/initial_distance_to_goal_per_env",
                per_env_init_distance,
                self._hist_step,
            )
            self._hist_step += 1
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new goal positions in the local environment frame and shift by each env origin.
        goal_min = torch.tensor(
            [self.cfg.goal_sample_x_range[0], self.cfg.goal_sample_y_range[0], self.cfg.goal_sample_z_range[0]],
            device=self.device,
        )
        goal_max = torch.tensor(
            [self.cfg.goal_sample_x_range[1], self.cfg.goal_sample_y_range[1], self.cfg.goal_sample_z_range[1]],
            device=self.device,
        )
        self._desired_pos_w[env_ids] = goal_min + (goal_max - goal_min) * torch.rand(len(env_ids), 3, device=self.device)
        self._desired_pos_w[env_ids] += self._terrain.env_origins[env_ids]
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        init_distance = torch.linalg.norm(
            self._desired_pos_w[env_ids, :2] - default_root_state[:, :2], dim=1
        )
        self._episode_init_distance[env_ids] = init_distance

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

    def _debug_vis_callback(self, event):
        # update the markerah cool. s
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
