import copy
import sys

import numpy as np
import torch
import tqdm
from skrl.trainers.torch import Trainer
from skrl.trainers.torch.sequential import SequentialTrainer
from torch.utils.tensorboard import SummaryWriter

META_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 128,  # timesteps per iteration inside inner loop
}

TRAJ_DICT = {
    0: "hover",
    1: "straight",
    2: "circle",
}
# META_TIMESTEPS =


class MetaTrainer(SequentialTrainer):
    def __init__(self, env, agents, agents_scope=None, cfg=None):
        _cfg = copy.deepcopy(META_TRAINER_DEFAULT_CONFIG)
        print(f"MetaTrainer cfg: {cfg}")
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)
        # self.agents= self.agents  # assuming single-agent setup
        # self.meta_lr = cfg.get("meta_lr", 0.08)

        self.tasks_per_meta = cfg.get("tasks_per_meta", 3)
        self.inner_loops = cfg.get("inner_loops", 1)
        self.meta_iterations = cfg.get("meta_iterations", 70)

        self.meta_lr = agents.cfg.get("meta_learning_rate", 0.082)
        self.train_base_flag = agents.cfg.get("train_base_policy", False)
        self.curriculum = agents.cfg.get("curriculum", "randomized")

        log_dir = f"{self.agents.cfg['experiment']['directory']}/{self.agents.cfg['experiment']['experiment_name']}"
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"logdir:{log_dir}")  # assuming single-agent setup

    def snapshot_params(self):
        policy = {k: v.detach().clone() for k, v in self.agents.policy.state_dict().items()}
        value = {k: v.detach().clone() for k, v in self.agents.value.state_dict().items()}
        return policy, value

    @torch.no_grad()
    def load_params_(self, state, both=True):
        self.agents.policy.load_state_dict(state, strict=True)
        if both:
            self.agents.value.load_state_dict(state, strict=True)

    @torch.no_grad()
    def add_scaled_(self, acc, delta, scale):
        for k in acc:
            acc[k].add_(scale * delta[k])

    def _log_layerwise_delta_stats(self, delta, meta_step):
        for k, v in delta.items():
            layer_delta_norm = torch.norm(v).item()
            self.writer.add_scalar(f"Meta/Delta_Norm/{k}", layer_delta_norm, meta_step)

    def train(self):
        # super().train()
        self.meta_step = 0
        #
        if self.train_base_flag:
            print(f"Training base policy before meta-learning-----------------------------------------------------")
            self._train_inner(timestep=None, train_base=True)

        for meta_it in range(self.meta_iterations):
            print(f"Using curriculum type :{self.curriculum}-------------------------------------------------")
            base_policy, _ = self.snapshot_params()
            avg_delta = {k: torch.zeros_like(v) for k, v in base_policy.items()}

            for j in range(self.tasks_per_meta):
                print(f"[Meta Iteration {meta_it+1}/{self.meta_iterations}] Task {j+1}/{self.tasks_per_meta}")
                self.env.reset()
                self.load_params_(base_policy, both=True)
                # self.load_params_(adapted)               # start from θ

                for i in range(self.inner_loops):
                    print(f" Inner Loop {i+1}/{self.inner_loops}")
                    print(f"training from timestep: {self.initial_timestep} to {self.timesteps}")
                    self._train_inner(timestep=None)

                # self.eval()
                self.meta_step += 1
                adapted_policy, adapted_value = self.snapshot_params()
                # ----------------------------------------------------------------------
                # new_weights = {k: adapted_policy[k] for k in adapted_policy}
                # self.load_params_(new_weights, both=True)
                # ------------------------------------------------------------------------------
                delta = {k: adapted_policy[k] - base_policy[k] for k in base_policy}
                self.add_scaled_(avg_delta, delta, 1.0 / self.tasks_per_meta)

                total_delta = sum((v**2).sum().item() for v in delta.values())
                self.writer.add_scalar(f"Meta/Total_Delta", total_delta, self.meta_step)

                self._log_layerwise_delta_stats(delta, self.meta_step)

            new_weights = {k: base_policy[k] + self.meta_lr * avg_delta[k] for k in base_policy}

            self.load_params_(new_weights, both=True)

            # meta evaluation with the learned parameters
            print(f"[Meta Iteration {meta_it+1}/{self.meta_iterations}] Meta Evaluation")
            self.eval()

    def _get_traj_type(self):
        if self.curriculum == "randomized":
            return TRAJ_DICT[np.random.randint(0, self.tasks_per_meta)]
        elif self.curriculum == "sequential":
            return TRAJ_DICT[self.meta_step % self.tasks_per_meta]
        elif self.curriculum == "structured":
            return TRAJ_DICT[(self.meta_step // (self.meta_iterations // self.tasks_per_meta)) % self.tasks_per_meta]

    def _train_base_policy(self):
        self.env.trajectory_generator.randomise_task = True
        self.env.trajectory_generator.trajectory_type = "hover"
        self.env.trajectory_generator.reset(time=torch.zeros(self.env.num_envs, device=self.env.device))
        print(f"Training base policy on hover trajectory-------------------------------------------------")

    def _set_traj_type_and_reset_env(self):
        self.env.trajectory_generator.randomise_task = True
        self.env.trajectory_generator.trajectory_type = self._get_traj_type()
        self.env.trajectory_generator.reset(time=torch.zeros(self.env.num_envs, device=self.env.device))
        # remove lemniscate traj type
        while self.env.trajectory_generator.trajectory_type == "lemniscate":
            self.env.trajectory_generator.reset(time=torch.zeros(self.env.num_envs, device=self.env.device))

    def _train_inner(self, timestep, train_base=False) -> None:
        # pre-interaction
        print(f"learning rate :{self.meta_lr}")
        self.agents.set_running_mode("train")
        # reset env

        if train_base and self.train_base_flag:
            self._train_base_policy()
            # self.train_base_flag = 1
            print(f"Base Policy Training-------------------------------------------------")
        else:
            self.train_base_flag = False
            self._set_traj_type_and_reset_env()

        print(
            "Inner loop training with :"
            f" {self.env.trajectory_generator.trajectory_type}-------------------------------------------------"
        )
        # print(f"Inner Loop: Resetting env with random traj type: {self.env.trajectory_generator.trajectory_type}")

        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # log all the agent data to tensorboard
            logs = self.agents.tracking_data
            for key, value in logs.items():
                # print(f"Logging {key}: {value}")
                if self.writer:
                    # exclude min and max values
                    if not key.endswith("min") or not key.endswith("max"):
                        self.writer.add_scalar(
                            f"Policy/{key}", np.mean(value), int(self.meta_step * self.timesteps + timestep)
                        )

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def eval(self) -> None:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """

        # evaluate on lemniscate trajectory
        self.env.trajectory_generator.randomise_family = False
        self.env.trajectory_generator.randomise_task = True
        self.env.trajectory_generator.trajectory_type = "lemniscate"
        self.env.trajectory_generator.reset(time=torch.zeros(self.env.num_envs, device=self.env.device))

        print(
            "Eval: Resetting env with traj type:"
            f" {self.env.trajectory_generator.trajectory_type}-------------------------------------------------"
        )
        # set running mode
        timesteps = 300
        print(f"inside eval, timesteps: {timesteps}")
        meta_reward = 0.0
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(range(0, timesteps), disable=self.disable_progressbar, file=sys.stdout):
            print(f"Eval Timestep: {timestep+1}/{timesteps}", end="\r")

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=timesteps)

            with torch.no_grad():
                # print(f'torch.no_grad: {states.shape}')
                # compute actions
                actions = self.agents.act(states, timestep=timestep, timesteps=timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                # print(f'rewards: {rewards}\t {rewards.shape}\t{rewards.mean().item()}')
                meta_reward += rewards.mean().item()
                self.writer.add_scalar(
                    "Meta/Step_Reward",
                    rewards.mean().item(),
                    int(((self.meta_step // self.tasks_per_meta) - 1) * timesteps + timestep),
                )
                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                # for agent, scope in zip(self.agents, self.agents_scope):
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

                        # log all the agent data to tensorboard

            logs = self.agents.tracking_data
            for key, value in logs.items():
                # print(f"Logging {key}: {value}")
                if self.writer:
                    # exclude min and max values
                    if not key.endswith("min") or not key.endswith("max"):
                        self.writer.add_scalar(
                            f"Meta/{key}", np.mean(value), int(self.meta_step * self.timesteps + timestep)
                        )

            # post-interaction
            super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=timesteps)

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states

        print(f"Meta-Eval Reward: {meta_reward }")
        self.writer.add_scalar("Meta/Eval_Reward", meta_reward, self.meta_step)
        return meta_reward
