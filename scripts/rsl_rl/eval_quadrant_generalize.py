# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate quadrant generalisation for a symmetry-augmented quadrotor policy.

Loads a policy trained with ``Quadcontrol-QuadrantGoal-SymmAugMirror-v0``
(goals sampled only from the 1st quadrant) and rolls it out in
``Quadcontrol-Full3DGoal-SymmAugMirror-v0`` (goals spanning all four quadrants
of the xy-plane). Reports the mean final distance-to-goal, the success rate and
a per-quadrant breakdown.

Usage:
    python scripts/rsl_rl/eval_quadrant_generalize.py --task Quadcontrol-Full3DGoal-SymmAugMirror-v0 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate quadrant generalisation.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Quadcontrol-Full3DGoal-SymmAugMirror-v0", help="Name of the task.")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to evaluate per environment.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from collections import defaultdict

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import QuadControl.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner


def _quadrant_index(xy: torch.Tensor) -> int:
    """Map a goal (x, y) to a quadrant index: 0..3 (x>=0,y>=0 -> 0; x<0,y>=0 -> 1; ...)."""
    x = float(xy[0])
    y = float(xy[1])
    if x < 0 and y >= 0:
        return 1
    if x < 0 and y < 0:
        return 2
    if x >= 0 and y < 0:
        return 3
    return 0


def main():
    """Evaluate the policy across all four goal quadrants."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # locate the checkpoint produced by the training run (shared experiment_name)
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Checkpoint: {resume_path}")

    # create the full-3D evaluation environment
    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(resume_path), "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load the trained policy
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # bookkeeping for per-quadrant statistics
    quad_final_dist = defaultdict(list)
    quad_success = defaultdict(int)
    quad_count = defaultdict(int)

    obs, _ = env.get_observations()
    inner = env.unwrapped
    step_count = 0

    print("Evaluating...")
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

        # The env stores, in `_reset_idx`, the goal (local frame) and final
        # distance of every episode that terminated on this step. Read them
        # before the next step overwrites them.
        terminated = (dones.view(-1) > 0).nonzero(as_tuple=False).view(-1)
        for i in terminated.tolist():
            goal_local = inner._eval_goal_local[i]
            final_dist = float(inner._eval_final_dist[i])
            qi = _quadrant_index(goal_local[:2])
            quad_final_dist[qi].append(final_dist)
            quad_count[qi] += 1
            if final_dist < 0.3:
                quad_success[qi] += 1

        step_count += 1
        if step_count >= args_cli.num_episodes * inner.max_episode_length:
            break

    # report
    quad_names = ["Q1 (x>=0,y>=0)", "Q2 (x<0,y>=0)", "Q3 (x<0,y<0)", "Q4 (x>=0,y<0)"]
    print("=" * 70)
    print("  Quadrant Generalisation Report")
    print("=" * 70)
    all_dists = []
    total_success = 0
    total_count = 0
    for qi in range(4):
        dists = quad_final_dist.get(qi, [])
        count = quad_count.get(qi, 0)
        success = quad_success.get(qi, 0)
        all_dists.extend(dists)
        total_success += success
        total_count += count
        if count > 0:
            mean_dist = sum(dists) / len(dists)
            print(
                f"  {quad_names[qi]:18s}  episodes={count:4d}  "
                f"mean_final_dist={mean_dist:.3f} m  success={success/count*100:.1f}%"
            )
        else:
            print(f"  {quad_names[qi]:18s}  episodes=0")
    if total_count > 0:
        print("-" * 70)
        print(f"  {'Overall':18s}  episodes={total_count:4d}  "
              f"mean_final_dist={sum(all_dists)/len(all_dists):.3f} m  "
              f"success={total_success/total_count*100:.1f}%")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
