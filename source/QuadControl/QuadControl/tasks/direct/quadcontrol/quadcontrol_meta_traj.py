# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
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

from isaaclab.markers import CUBOID_MARKER_CFG, RED_ARROW_X_MARKER_CFG  # isort: skip
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass
# from isaacsim.util.debug_draw import _debug_draw


from .quadcontrol_env_cfg import QuadcontrolEnvCfg
from ..trajectory_generator import TrajectoryGenerator

class QuadcontrolEnv(DirectRLEnv):
    cfg: QuadcontrolEnvCfg

    def __init__(self, cfg: QuadcontrolEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = 9 + 3 * cfg.num_lookahead  # lin_vel, ang_vel, gravity + desired_pos for each lookahead step

        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # print(f"Actions shape:, {self._actions.shape} \n  num_envs: {self.num_envs}, action_space: {self.single_action_space}")
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self.num_lookahead = self.cfg.num_lookahead    
        print(f"Number of lookahead waypoints: {self.num_lookahead}")    
        # Goal position / trajectory position - dynamically updated
        # self.cfg.observation_space = 9 + 3 * self.num_lookahead  # lin_vel, ang_vel, gravity + desired_pos for each lookahead step
        
        num_pos_hist = 10

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.current_point = torch.zeros(self.num_envs, 3, device=self.device)
        if self.num_lookahead>1:
            self._desired_pos_w = torch.zeros(self.num_envs, self.num_lookahead, 3, device=self.device)
            self.current_point = torch.zeros(self.num_envs, self.num_lookahead, 3, device=self.device)
     
        
        self._pos_history = torch.zeros(self.num_envs, num_pos_hist, 3, device=self.device)
        self._hist_idx = 0
        
        # self.max_traj_length = 500
        # # for viaualization of reference and tracking 
        # self._reference_pos_w = torch.zeros(self.num_envs, self.max_traj_length, 3, device=self.device)
        # self._tracking_pos_w = torch.zeros(self.num_envs, self.max_traj_length, 3, device=self.device)
        # self.traj_step = torch.zeros(self.num_envs, dtype=torch.int32)
        self.tilt_angle = torch.zeros(self.num_envs, device=self.device)
        # self._desired_pos_w = torch.rand(self.num_envs, 3, device=self.device)
        # trajectory generator
        self.trajectory_type = self.cfg.trajectory_type
        # print(f"trhust tot weight: {self.cfg.thrust_to_weight} \t moment scale: {self.cfg.moment_scale}")
        # self.randomise_family = cfg.randomise_family
        # self.randomise_task = cfg.randomise_task

        print(f"Using trajectory type: {self.trajectory_type}")
        self.time = torch.zeros(self.num_envs, device=self.device)
        

        self.trajectory_generator = TrajectoryGenerator(device=self.device, 
                                                        # randomise_task=cfg.randomise_task, 
                                                        trajectory_type=cfg.trajectory_type,
                                                        randomise_family=False,
                                                        randomise_task=True,
                                                        num_lookahead=self.num_lookahead,   
                                                        dt=self.step_dt,                                                     
                                                        )

        self.trajectory_generator.reset(self.time)
        # self._desired_pos_w = self.robot.data.root_pos_w[:, :3]
       # self.eval_log = self.cfg.get{"eval_log", False} 
        self.eval_log = getattr(self.cfg, "eval_log", False)

        if self.eval_log: 
            self.eval_log_dir = Path(self.cfg.log_dir) / "eval_logs"
            self.eval_log_dir.mkdir(parents=True, exist_ok=True)
            print(f"Evaluation logging enabled. Log dir: {self.eval_log_dir}")
            
            self._traj_log = []
        self.died = [False]
        self.time_out = [False]
        self.eps=0
        # # Logging
        # self.trajectory_generator = TrajectoryGenerator(
        #     device=self.device,
        #     # randomise_family=self.randomise_family,
        #     # randomise_task=self.randomise_task,
        #     trajectory_type=self.trajectory_type,
        # )

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                # "bonus",
            ]
        }
        self._episode_actions = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "thrust",
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
        
        # tracking_vectors = {
        #     key: torch.zeros(self.num_envs, 3, self.episode_length_buf ,dtype=torch.float, device=self.device)
        #     for key in [
        #         "robot_pos"
        #         "desired_pos",
        #     ]
        # }

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)


        # self._draw = _debug_draw.acquire_debug_draw_interface()


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
        self._actions = actions.clone().clamp(-1.0, 1.0)
        # print(len(self._actions), self._actions.shape, self._actions.dtype)
        # self.tilt_angle = 
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]


        forces = {
            "thrust": self._thrust[:, 0, 2],
            "moment_x": self._moment[:, 0, 0],
            "moment_y": self._moment[:, 0, 1],
            "moment_z": self._moment[:, 0, 2],
        }
        for key in self._episode_actions:
            self._episode_actions[key] += forces[key]

        self._episode_errors["control_effort"] += torch.sum(self._actions ** 2, dim=1)
        self.post_physics_step()
        #     "robot_pos": self._robot.data.root_pos_w,)


    def post_physics_step(self):
        if not self.eval_log:
            return  # no logging in training
        
        # collect state for env 0
        # print(self.episode_length_buf[0], self.max_episode_length)
        rs = self._robot.data.root_state_w  # (num_envs, 13): pos, quat, lin_vel, ang_vel[web:3]
        pos = rs[0, 0:3]
        quat = rs[0, 3:7]
        lin_vel = rs[0, 7:10]
        ang_vel = rs[0, 10:13]
        ref_pos = self._desired_pos_w[0]

        self._traj_log.append(
            dict(
                t=float(self.episode_length_buf[0] * self.step_dt),
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                qx=float(quat[0]), qy=float(quat[1]), qz=float(quat[2]), qw=float(quat[3]),
                vx=float(lin_vel[0]), vy=float(lin_vel[1]), vz=float(lin_vel[2]),
                wx=float(ang_vel[0]), wy=float(ang_vel[1]), wz=float(ang_vel[2]),
                xref=float(ref_pos[0]), yref=float(ref_pos[1]), zref=float(ref_pos[2]),
                f_z=float(self._thrust[0, 0, 2]),f_x=float(self._thrust[0, 0, 0]),f_y=float(self._thrust[0, 0, 1]),
                m_x=float(self._moment[0, 0, 0]),m_y=float(self._moment[0, 0, 1]),m_z=float(self._moment[0, 0, 2]),
                tilt_angle=float(self.tilt_angle[0]), 
            )
        )

        done_max = self.died[0] or self.time_out[0]
        if done_max:
            # self.died = self.episode_length_buf[0]
            # print("evaluation episode done, saving trajectory log...")
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
        desired_pos_b = torch.zeros_like(self._desired_pos_w)
        # transoform for each nest waypoints
        if self.num_lookahead>1:
            for i in range(self.num_lookahead):
                desired_pos_b[:, i, :], _ = subtract_frame_transforms(
                    self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w[:, i, :]
                )
        else:
            desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
            )
        # next_point = self.trajectory_generator.generate_trajectory(self.time)
        next_point = self.trajectory_generator.generate_multiple_goals(self.time, num_goals=self.num_lookahead)

        # store reference and tracking positions for visualization
        # tracking_vectors = {
        #     "robot_pos": self._robot.data.root_pos_w,
        #     "desired_pos": self._desired_pos_w,
        # }

        # print(delta)
        # print(f"{next_point.shape, self.current_point.shape, self.time.shape}")
        # print(f" \t Current position: {self._desired_pos_w}")
        # print robot position
        # print(f"robot position: {self._robot.data.root_pos_w[10]}")
        # print(f"desired position before update: {self._desired_pos_w[10]} \t current point: {self.current_point[10]}\t time: {self.time[10]}")
        if self.trajectory_type != "hover":
            # print next point and current point for env 1
            # print(f"next point: {next_point.shape} \t current point: {self.current_point.shape}\t time: {self.time[10]}")
            delta = next_point - self.current_point
            self.current_point = next_point
            self._desired_pos_w += delta  # broadcast delta to all envs and lookahead steps
        else:
            delta = next_point
            self._desired_pos_w[(self.time==0).nonzero(as_tuple=True)[0], :2] += delta[(self.time==0).nonzero(as_tuple=True)[0], :2]
            self._desired_pos_w[(self.time==0).nonzero(as_tuple=True)[0], 2] = delta[(self.time==0).nonzero(as_tuple=True)[0], 2]
        # print all
        # print(f"desired position after update: {self._desired_pos_w[10]} \t delta: {delta[10]}")
        # print(f"next point: {next_point[10]} \t current point: {self.current_point[10]} \t time: {self.time[10]}")
        # self._desired_pos_w += delta
        # print(f"Desired position: {self._desired_pos_w[0]} \t delta: {delta[0]}")
        
        if self.num_lookahead>1:
            desired_pos_b = desired_pos_b.reshape(self.num_envs, -1)  # flatten lookahead waypoints into a single vector for observation

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        self.time += self.step_dt

        self._pos_history[:, self._hist_idx, :] = self._robot.data.root_pos_w[:, :3]
        self._hist_idx = (self._hist_idx + 1) % self._pos_history.shape[1]
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        # if shape is 3 dim 
        if self._desired_pos_w.shape[1] == self.num_lookahead:
            distance_to_goal = torch.linalg.norm(self._desired_pos_w[:, 0, :] - self._robot.data.root_pos_w[:, :3], dim=1)
        else:
            distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        # goal_reached = distance_to_goal < 0.1
        # bonus = torch.where(goal_reached, torch.tensor(2.0), torch.tensor(0.0))
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,    # self._tracking_pos_w[:self.traj_step] = 0.0
        # self._reference_pos_w[:self.traj_step] = 0.0
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            # "bonus": bonus * self.step_dt
        }
    
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
    
        self._episode_errors["tracking_cost"] += distance_to_goal
        
        for key, value in rewards.items():
            self._episode_sums[key] += value 
        return reward
    

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        self.died, self.time_out = died, time_out
        return died, time_out

    # call this with env_ids=None to reset all robot instances :TODO
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        if self._desired_pos_w.shape[1] == self.num_lookahead:
            final_distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[:, 0, :] - self._robot.data.root_pos_w[:, :3], dim=1
            ).mean()
        else:
            final_distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
            ).mean()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        
        for key in self._episode_actions.keys():
            episodic_action_avg = torch.mean(self._episode_actions[key][env_ids])
            extras["Action/" + key] = episodic_action_avg / self.max_episode_length_s
            self._episode_actions[key][env_ids] = 0.0

        for key in self._episode_errors.keys():
            episodic_error_avg = torch.mean(self._episode_errors[key][env_ids])
            episodic_error_std = torch.std(self._episode_errors[key][env_ids])
            extras["Error/" + key] = episodic_error_avg / self.max_episode_length_s
            extras["Error/" + key + "_std"] = episodic_error_std
            self._episode_errors[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids])
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids])
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal
        # extras["Error/tracking_cost"] = final_distance_to_goal
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids])
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids])
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.time[env_ids] = 0.0
        # print(f"resetting envs: , {env_ids}, time reset to zero.: {self.time.any() == 0.0}")

        self._actions[env_ids] = 0.0
        # Reset robot state
        self.current_point[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Sample new commands
        # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2])
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # self._desired_pos_w[env_ids, 2] = torch.ones_like(self._desired_pos_w[env_ids, 2])
        
        # --------------whether to change the task parameters on reset? --------------
        # self.trajectory_generator.reset(self.time)
        # let the initial position be the starting point of the trajectory
        if self.num_lookahead>1:
            for i in range(self.num_lookahead):
                self._desired_pos_w[env_ids, i, :] = default_root_state[:, :3]
        else:
            self._desired_pos_w[env_ids] = default_root_state[:, :3]

        self._pos_history[env_ids] = 0.0

        # self._desired_pos_w[env_ids, :3] = default_root_state[:, :3]
        # self.traj_step[env_ids] = 0

        # self.current_point[env_ids] += self._desired_pos_w[env_ids]


    # def _update_traj_markers(self):
    #     self._reference_pos_w[:, self.traj_step, :] = self._desired_pos_w
    #     self._tracking_pos_w[:, self.traj_step, :] = self._robot.data.root_pos_w
    #     self.traj_step.add_(1)

        # if self.traj_step >= self.max_traj_length:
            # self.traj_step = 0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer = []
                self.track_pos_visualizer = []
                track_pos = list()
                marker_cfg = CUBOID_MARKER_CFG.copy()
                track_marker_cfg = CUBOID_MARKER_CFG.copy()
                # force_marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                track_marker_cfg.markers["cuboid"].size = (0.02, 0.02, 0.02)

                marker_cfg.markers["cuboid"].color = (0.0, 1.0, 0.0, 0.8)  # green
                track_marker_cfg.markers["cuboid"].color = (1.0, 0.0, 0.0, 0.8)  # red
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                for i in range(10):
                    track_marker_cfg = self._make_marker_cfg(name=f"tracking_position_{i}", color=(1.0, 0.0, 0.0, 0.8), size=(0.02, 0.02, 0.02))
                    self.track_pos_visualizer.append(VisualizationMarkers(track_marker_cfg))
                    
                # print(f"in _set_debug_vis_impl: {self._desired_pos_w.shape}")

                if self.num_lookahead>1:
                    for i in range(self.num_lookahead):
                        marker_cfg = CUBOID_MARKER_CFG.copy()
                        marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                        marker_cfg.markers["cuboid"].color = (0.0, 1.0, 0.0, 0.8)  # green
                        marker_cfg.prim_path = f"/Visuals/Command/goal_position_{i}"
                        self.goal_pos_visualizer.append(VisualizationMarkers(marker_cfg))

                else:
                    self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
                # self.track_pos_visualizer = VisualizationMarkers(track_marker_cfg)
                # self.vel_marker_cfg = VisualizationMarkers(force_marker_cfg)
                # self.robot_velocity_visualizer = VisualizationMarkers(vel_marker_cfg)
            # set their visibility to true
            if self.num_lookahead>1:
                for i in range(self.num_lookahead):
                    self.goal_pos_visualizer[i].set_visibility(True)
            if self.num_lookahead==1:
                self.goal_pos_visualizer.set_visibility(True)

            for i in range(10):
                self.track_pos_visualizer[i].set_visibility(True)
            # self.track_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                for i in range(self.num_lookahead):
                    self.goal_pos_visualizer[i].set_visibility(False)
            self.track_pos_visualizer.set_visibility(False)
    
    def _make_marker_cfg(self, name, color, size):
        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = size
        marker_cfg.markers["cuboid"].color = color
        marker_cfg.prim_path = f"/Visuals/Command/{name}"
        return marker_cfg

    def _debug_vis_callback(self, event):
        # update the markers
        # self._tracking_pos_w[:self.traj_step] = 0.0
        # self._reference_pos_w[:self.traj_step] = 0.0
        # self._update_traj_markers()
        # self._reference_pos_w.append(self._desired_pos_w)
        if self.num_lookahead>1:
            for i in range(self.num_lookahead):
                pt = self._desired_pos_w[:, i, :]
                # pt = pt.squeeze(1)
                self.goal_pos_visualizer[i].visualize(pt)
        else:
            self.goal_pos_visualizer.visualize(self._desired_pos_w)

        history = self._get_history()
        trail_pts = history[:, :, :].reshape(-1, 3)

        for i in range(min(10, history.shape[1])):
            self.track_pos_visualizer[i].visualize(trail_pts[i:i+1])
        # self.track_pos_visualizer.visualize(self._robot.data.root_pos_w)
        # self.vel_marker_cfg.visualize(
        #     self._robot.data.root_pos_w,
        #     self._robot.data.root_lin_vel_w * 0.1,  # scale down for visualization
        # )
        # self.track_pos_visualizer.visualize(self._reference_pos_w)

    def _get_history(self):
        idx = [(i + self._hist_idx) % self._pos_history.shape[1] for i in range(self._pos_history.shape[1])]
        return self._pos_history[:, idx, :]
