import copy
import itertools
import sys

import numpy as np
import torch
import tqdm
from skrl.trainers.torch import Trainer
from skrl.trainers.torch.sequential import SequentialTrainer
from torch.optim import Adam
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

        self.tasks_per_meta = cfg.get("tasks_per_meta", 5)
        self.inner_loops = cfg.get("inner_loops", 2)
        self.meta_iterations = cfg.get("meta_iterations", 100)

        self.meta_lr = agents.cfg.get("meta_learning_rate", 1e-2)
        self.train_base_flag = agents.cfg.get("train_base_policy", False)
        self.curriculum = agents.cfg.get(
            "curriculum", "none"
        )  # options: "none", "randomized", "sequential", "structured"

        # for meta-optimization
        self._meta_optim = Adam(
            itertools.chain(self.agents.policy.parameters(), self.agents.value.parameters()), lr=self.meta_lr
        )

        log_dir = f"{self.agents.cfg['experiment']['directory']}/{self.agents.cfg['experiment']['experiment_name']}"
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"logdir:{log_dir}")  # assuming single-agent setup

    def snapshot_params(self):
        policy = {k: v.detach().clone() for k, v in self.agents.policy.state_dict().items()}
        value = {k: v.detach().clone() for k, v in self.agents.value.state_dict().items()}
        return policy, value

    @torch.no_grad()
    def load_params_(self, state, both=True):
        base_policy, base_value = state
        self.agents.policy.load_state_dict(state, strict=True)
        if both:
            self.agents.value.load_state_dict(state, strict=True)

    # def train(self):
    #     # super().train()
    #     self.meta_step = 0
    #     #
    #     if self.train_base_flag:
    #         print(f"Training base policy before meta-learning-----------------------------------------------------")
    #         self._train_inner(timestep=None, train_base=True)

    #     for meta_it in range(self.meta_iterations):

    #         # initialize meta-gradient accumulators
    #         self._meta_optim.zero_grad()

    #         print(f"Using curriculum type :{self.curriculum}-------------------------------------------------")

    #         meta_loss = 0.0

    #         for j in range(self.tasks_per_meta):
    #             print(f"[Meta Iteration {meta_it+1}/{self.meta_iterations}] Task {j+1}/{self.tasks_per_meta}")
    #             self.env.reset()

    #             # inner loop training
    #             for i in range(self.inner_loops):
    #                 print(f" Inner Loop {i+1}/{self.inner_loops}")
    #                 print(f"training from timestep: {self.initial_timestep} to {self.timesteps}")
    #                 # inner-loop training with the current policy parameters
    #                 # self._train_inner(timestep=None)
    #                 support_rollout = self._collect_rollout(timesteps=self.timesteps)
    #                 support_batch = self.agents.build_batch_from_rollout(support_rollout)

    #                 fpolicy, fvalue, logs = self.agents.differentiable_update(
    #                     support_batch,
    #                     inner_steps=600,
    #                     inner_lr=self.meta_lr,
    #                     track_higher_grads=True,
    #                       )

    #                 query_rollout = self._collect_rollout_with_functional_policy(
    #                         self.timesteps, fpolicy, fvalue
    #                     )
    #                 query_batch = self.agents.build_batch_from_rollout(query_rollout)

    #                 query_loss, query_info = self.agents.compute_query_loss(
    #                         query_batch=query_batch,
    #                         policy=fpolicy,
    #                         value=fvalue,
    #                     )

    #                 (query_loss / self.tasks_per_meta).backward()
    #                 meta_loss += query_loss.detach().item()

    #             self._meta_optim.step()
    #             self.meta_step += 1

    #             # compute meta-loss (negative of eval reward)

    #             self.writer.add_scalar(f"Meta/Loss", meta_loss, self.meta_step)

    #         # meta evaluation with the learned parameters
    #         print(f"[Meta Iteration {meta_it+1}/{self.meta_iterations}] Meta Evaluation")
    #         self.eval()

    def train(self):
        self.meta_step = 0

        if self.train_base_flag:
            self._train_inner(timestep=None, train_base=True)

        for meta_it in range(self.meta_iterations):
            print(f"[Meta Iteration {meta_it+1}/{self.meta_iterations}]")

            meta_loss = self._meta_step_maml(track_higher_grads=True)

            self.meta_step += 1
            self.writer.add_scalar("Meta/Loss", meta_loss, self.meta_step)

            print(f"Meta loss: {meta_loss:.6f}")

            self.eval()

    def _collect_rollout_with_functional_policy(self, timesteps, fpolicy, fvalue):
        rollout = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "terminated": [],
            "truncated": [],
            "log_prob": [],
            "values": [],
        }

        states, infos = self.env.reset()

        for t in range(timesteps):
            with torch.no_grad():
                proc_states = self.agents._state_preprocessor(states)

                actions, log_prob, _ = fpolicy.act({"states": proc_states}, role="policy")
                values, _, _ = fvalue.act({"states": proc_states}, role="value")

                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            rollout["states"].append(states.detach())
            rollout["actions"].append(actions.detach())
            rollout["rewards"].append(rewards.detach())
            rollout["next_states"].append(next_states.detach())
            rollout["terminated"].append(terminated.detach())
            rollout["truncated"].append(truncated.detach())
            rollout["log_prob"].append(log_prob.detach())
            rollout["values"].append(values.detach())

            states = next_states

        for k in rollout:
            rollout[k] = torch.stack(rollout[k], dim=0)

        return rollout

    def _meta_step_maml(self, track_higher_grads=True):
        self._meta_optim.zero_grad()
        meta_loss = 0.0

        for task_idx in range(self.tasks_per_meta):
            # inner loop training for task adaptation
            self._set_traj_type_and_reset_env()

            for inner_loop in range(self.inner_loops):
                print(f" Task {task_idx+1}/{self.tasks_per_meta} - Inner Loop {inner_loop+1}/{self.inner_loops}")

                support_rollout = self._collect_rollout(self.timesteps)
                # support_batch = self.agents.
                support_batch = self.agents.build_batch_from_rollout(support_rollout)

                # for batch in [support_batch]:  # if you want to use multiple batches from the same rollout
                for epoch in range(4):  # you can increase this for better adaptation
                    fpolicy, fvalue, support_logs = self.agents.differentiable_inner_update(
                        batch=support_batch,
                        inner_steps=self.inner_loops,
                        inner_lr=self.meta_lr,
                        track_higher_grads=track_higher_grads,
                        # epochs=1,
                    )

                self.writer.add_scalar(f"Meta/Support_Loss/Entropy", support_logs["entropy_loss"], self.meta_step)
                self.writer.add_scalar(f"Meta/Support_Loss/Value", support_logs["value_loss"], self.meta_step)
                self.writer.add_scalar(f"Meta/Support_Loss/Policy", support_logs["policy_loss"], self.meta_step)

            query_rollout = self._collect_rollout_with_functional_policy(self.timesteps, fpolicy, fvalue)

            query_batch = self.agents.build_batch_from_rollout(query_rollout)

            query_loss, query_info = self.agents.compute_query_loss(
                query_batch=query_batch,
                policy=fpolicy,
                value=fvalue,
            )

            (query_loss / self.tasks_per_meta).backward()
            meta_loss += query_loss.detach().item()

        # meta-optimization step
        self._meta_optim.step()
        return meta_loss / self.tasks_per_meta

    def _get_traj_type(self):
        if self.curriculum == "randomized":
            return TRAJ_DICT[np.random.randint(0, 3)]
        elif self.curriculum == "sequential":
            return TRAJ_DICT[self.meta_step % 3]
        elif self.curriculum == "structured":
            return TRAJ_DICT[(self.meta_step // (self.meta_iterations // 3)) % 3]
        elif self.curriculum == "none":
            return "hover"

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

    def _collect_rollout(self, timesteps, policy_override=None):
        rollout = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "terminated": [],
            "truncated": [],
            "log_prob": [],
            "values": [],
        }

        states, infos = self.env.reset()

        for t in range(timesteps):
            with torch.no_grad():
                if policy_override is None:
                    actions, log_prob, _ = self.agents.act(states, timestep=t, timesteps=timesteps)
                    values, _, _ = self.agents.value.act(
                        {"states": self.agents._state_preprocessor(states)}, role="value"
                    )
                else:
                    proc_states = self.agents._state_preprocessor(states)
                    actions, log_prob, _ = policy_override.act({"states": proc_states}, role="policy")
                    values, _, _ = self.agents.value.act({"states": proc_states}, role="value")

                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            rollout["states"].append(states.detach())
            rollout["actions"].append(actions.detach())
            rollout["rewards"].append(rewards.detach())
            rollout["next_states"].append(next_states.detach())
            rollout["terminated"].append(terminated.detach())
            rollout["truncated"].append(truncated.detach())
            rollout["log_prob"].append(log_prob.detach())
            rollout["values"].append(values.detach())

            states = next_states

        for k in rollout:
            rollout[k] = torch.stack(rollout[k], dim=0)

        return rollout

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
        self.env.trajectory_generator.trajectory_type = "hover"
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
