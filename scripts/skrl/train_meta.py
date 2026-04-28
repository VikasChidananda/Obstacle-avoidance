# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train quadrotor meta-RL (MAML/Reptile) with skrl."""

import argparse
import os
import random
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a quadrotor meta-RL policy with skrl")
parser.add_argument("--meta_algo", type=str, default="maml", choices=["maml", "reptile"])
parser.add_argument("--task", type=str, default=None, help="Gym task ID")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--video_interval", type=int, default=5000)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--experiment", type=str, default=None)

parser.add_argument("--meta_iterations", type=int, default=None)
parser.add_argument("--tasks_per_meta", type=int, default=None)
parser.add_argument("--inner_updates", type=int, default=None)
parser.add_argument("--query_updates", type=int, default=None)
parser.add_argument("--meta_lr", type=float, default=None)
parser.add_argument("--task_families", type=str, default="hover,straight,circle,lemniscate")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]] + [arg for arg in hydra_args if not arg.startswith("--/")]

import gymnasium as gym
import skrl
from packaging import version

SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. Install using: pip install skrl>={SKRL_VERSION}"
    )
    raise SystemExit(1)

import isaaclab_tasks  # noqa: F401
import QuadControl.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from QuadControl.utils.custom_runner import CustomRunner


def _default_task_id(meta_algo: str) -> str:
    return "Quadcontrol-MetaMAML-v0" if meta_algo == "maml" else "Quadcontrol-MetaReptile-v0"


def _cfg_entry_point(meta_algo: str) -> str:
    return "skrl_maml_cfg_entry_point" if meta_algo == "maml" else "skrl_reptile_cfg_entry_point"


task_id = args_cli.task or _default_task_id(args_cli.meta_algo)
agent_cfg_entry_point = _cfg_entry_point(args_cli.meta_algo)


@hydra_task_config(task_id, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    env_cfg.seed = agent_cfg["seed"]

    task_families = tuple(part.strip() for part in args_cli.task_families.split(",") if part.strip())
    if task_families:
        env_cfg.meta_task_families = task_families

    trainer_cfg = agent_cfg["trainer"]
    if args_cli.max_iterations is not None:
        trainer_cfg["timesteps"] = args_cli.max_iterations
    if args_cli.meta_iterations is not None:
        trainer_cfg["meta_iterations"] = args_cli.meta_iterations
    if args_cli.tasks_per_meta is not None:
        trainer_cfg["tasks_per_meta"] = args_cli.tasks_per_meta
    if args_cli.inner_updates is not None:
        trainer_cfg["inner_updates"] = args_cli.inner_updates
    if args_cli.query_updates is not None:
        trainer_cfg["query_updates"] = args_cli.query_updates
    if args_cli.meta_lr is not None:
        trainer_cfg["meta_learning_rate"] = args_cli.meta_lr

    trainer_cfg["close_environment_at_exit"] = False

    if args_cli.experiment is not None:
        agent_cfg["agent"]["experiment"]["directory"] = args_cli.experiment

    log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    run_name = f"meta_{args_cli.meta_algo}_torch_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg["agent"]["experiment"].get("experiment_name"):
        run_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"

    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name

    log_dir = os.path.join(log_root_path, run_name)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = CustomRunner(env, agent_cfg)

    if resume_path:
        runner.agent.load(resume_path)

    runner.run()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
