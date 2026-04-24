# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import random
import sys
from datetime import datetime

import optuna
from isaaclab.app import AppLauncher
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Quadcontrol-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC", "PPO-RNN", "custom"],
    help="The RL algorithm used for training the skrl agent.",
)
# determine which trajectory:
parser.add_argument(
    "--traj_type",
    type=str,
    default=None,
    help="specify the kind of trajectory - (hover, line, circle, lemnsicate, slalom)",
)

# experiment type for agent directories
parser.add_argument(
    "--experiment",
    type=str,
    default=None,
    help="Experiment type to be included in agent directory names",
)


# ------------------------------Hyper parameter optimization additions------------------------------
# Add optuna-specific CLI arguments
parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
parser.add_argument("--study-name", type=str, default="ppo_optimization", help="Name of the optimization study")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# https://github.com/isaac-sim/IsaacLab/issues/2729: to enbale livestream
# Filter out AppLauncher-injected args before Hydra sees them
sys.argv = [sys.argv[0]] + [arg for arg in hydra_args if not arg.startswith("--/")]

"""Rest everything follows."""

import os
import random
from datetime import datetime

import gymnasium as gym
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

import isaaclab_tasks  # noqa: F401
import QuadControl.tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from QuadControl.utils.custom_runner import CustomRunner

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def suggest_meta_params(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for PPO algorithm."""
    return {
        "meta_learning_rate": trial.suggest_float("learning_rate", 1e-5, 1, log=True),
        "timesteps": trial.suggest_int("timesteps", 10, 2400, step=160),
        # "epochs": trial.suggest_int("epochs", 5, 20),
        # "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        # "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        # "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        # "entropy_coef": trial.suggest_float("entropy_coef", 0.0, 0.01),
        # "value_loss_coef": trial.suggest_float("value_loss_coef", 0.5, 1.0),
        # "seed": trial.suggest_int("seed", 0, 10000),
    }


def objective(trial: optuna.Trial, env, base_agent_cfg: dict):
    """Optuna objective function for PPO optimization."""

    # Get suggested hyperparameters
    ppo_params = suggest_meta_params(trial)

    # Update agent config with suggested parameters
    agent_cfg = base_agent_cfg.copy()
    agent_cfg["agent"]["meta_learning_rate"] = ppo_params["meta_learning_rate"]
    agent_cfg["trainer"]["timesteps"] = ppo_params["timesteps"]

    # Create unique log directory for this trial
    log_root_path = os.path.join("logs", "skrl", args_cli.study_name)
    trial_dir = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(log_root_path, trial_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Update logging directory
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = trial_dir

    # env_cfg.seed = ppo_params["seed"]
    # Create environment
    env.reset()
    # Configure runner with smaller training duration for optimization
    # agent_cfg["trainer"]["timesteps"] = 1_000  # Reduce training time for optimization
    runner = CustomRunner(env, agent_cfg)
    runner.agent.set_running_mode("train")

    # Train and get results
    try:
        import torch
        import tqdm

        # take mean of three runs with same hyperparameters
        mean_reward = []

        runner.run()
        for i in range(5):
            # env.close()
            # runner.agent.load(os.path.join(log_dir, "checkpoints", "model_final.pt"))
            runner.agent.set_running_mode("eval")

            # reset environment
            obs, _ = env.reset()

            # simulate environment
            rewards = runner._trainer.eval()

            mean_reward.append(rewards)
            trial.report(mean_reward[-1], i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        final_reward = sum(mean_reward) / len(mean_reward)

        # for i, (iteration, evaluation) in enumerate(runner.run()):
        #     mean_reward = evaluation["eval/reward_mean"]
        #     rewards.append(mean_reward)

        #     # Report intermediate results for pruning
        #     trial.report(mean_reward, i)
        #     if trial.should_prune():
        #         raise optuna.TrialPruned()

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        final_reward = float("-inf")

        # finally:
        env.close()

    return final_reward


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    """Run Optuna hyperparameter optimization."""

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    env_cfg.trajectory_type = args_cli.traj_type

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # Configure the study
    study = optuna.create_study(
        study_name=args_cli.study_name,
        direction="maximize",
        sampler=TPESampler(),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    # Run optimization
    study.optimize(lambda trial: objective(trial, env, agent_cfg), n_trials=args_cli.n_trials, catch=(Exception,))

    # Print optimization results
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save study results
    study_dir = os.path.join("logs", "skrl", "optuna_studies")
    os.makedirs(study_dir, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(study_dir, f"{args_cli.study_name}_results.csv"))

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


# to run:
# python train_optuna.py --n-trials 100 --study-name "ppo_quad_optimization" --task Quadcontrol-Direct-v0
