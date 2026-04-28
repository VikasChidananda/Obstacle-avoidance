from __future__ import annotations

import copy
import sys

import numpy as np
import torch
import tqdm
from skrl.trainers.torch.sequential import SequentialTrainer
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from .task_sampler import QuadTask
from .meta_logging import MetaLogger


META_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 256,
    "meta_iterations": 50,
    "tasks_per_meta": 4,
    "inner_updates": 1,
    "query_updates": 1,
    "meta_learning_rate": 0.05,
    "eval_interval": 5,
    "eval_timesteps": 256,
    "log_task_batch": True,
    "log_task_params": True,
}


class BaseMetaTrainer(SequentialTrainer):
    """Shared utility trainer for first-order meta-RL with skrl PPO agents."""

    def __init__(self, env, agents, agents_scope=None, cfg=None):
        _cfg = copy.deepcopy(META_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []

        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        self.meta_iterations = int(_cfg.get("meta_iterations", 50))
        self.tasks_per_meta = int(_cfg.get("tasks_per_meta", 4))
        self.inner_updates = int(_cfg.get("inner_updates", 1))
        self.query_updates = int(_cfg.get("query_updates", 1))
        self.meta_lr = float(_cfg.get("meta_learning_rate", 0.05))
        self.eval_interval = int(_cfg.get("eval_interval", 5))
        self.eval_timesteps = int(_cfg.get("eval_timesteps", 256))
        self.log_task_batch = bool(_cfg.get("log_task_batch", True))
        self.log_task_params = bool(_cfg.get("log_task_params", True))

        log_dir = f"{self.agents.cfg['experiment']['directory']}/{self.agents.cfg['experiment']['experiment_name']}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.meta_logger = MetaLogger(self.writer, enabled=True)
        self.meta_step = 0

    def train(self):
        for meta_it in range(self.meta_iterations):
            base_state = self._snapshot_params()
            avg_delta = {k: torch.zeros_like(v) for k, v in base_state.items()}

            task_batch = self.env.unwrapped.task_sampler.sample_batch(self.tasks_per_meta)
            if self.log_task_batch:
                self.meta_logger.log_task_batch(task_batch, step=self.meta_step + 1)

            for task in task_batch:
                self._load_params(base_state)
                if self.log_task_params:
                    self.meta_logger.log_task(task, step=self.meta_step + 1)
                delta = self._collect_task_delta(task, base_state)
                self._add_scaled(avg_delta, delta, 1.0 / self.tasks_per_meta)

            grad_proxy_before = self._delta_to_grad_proxy(avg_delta, self.meta_lr)
            grad_norm_before = self._delta_norm(grad_proxy_before)

            new_state = {k: base_state[k] + self.meta_lr * avg_delta[k] for k in base_state}
            self._load_params(new_state)

            self.meta_step += 1
            update_delta = {k: new_state[k] - base_state[k] for k in base_state}
            self.writer.add_scalar("Meta/OuterStep", self.meta_step, self.meta_step)
            self.writer.add_scalar("Meta/DeltaNorm", self._delta_norm(avg_delta), self.meta_step)
            self.writer.add_scalar("Meta/UpdateNorm", self._delta_norm(update_delta), self.meta_step)
            self.writer.add_scalar("Meta/GradProxyNorm/BeforeOuterUpdate", grad_norm_before, self.meta_step)
            self._log_grad_proxy_by_module(grad_proxy_before, tag="BeforeOuterUpdate")

            # Re-estimate the meta-gradient proxy at updated weights on the same task batch.
            updated_state = self._snapshot_params()
            post_avg_delta = self._estimate_avg_task_delta(task_batch, updated_state)
            grad_proxy_after = self._delta_to_grad_proxy(post_avg_delta, self.meta_lr)
            grad_norm_after = self._delta_norm(grad_proxy_after)
            self.writer.add_scalar("Meta/GradProxyNorm/AfterOuterUpdate", grad_norm_after, self.meta_step)
            self._log_grad_proxy_by_module(grad_proxy_after, tag="AfterOuterUpdate")
            self._load_params(updated_state)

            if self.eval_interval > 0 and self.meta_step % self.eval_interval == 0:
                self.evaluate_meta()

    def evaluate_meta(self):
        self.agents.set_running_mode("eval")
        rewards = []
        rewards_by_family: dict[str, float] = {}

        for family in self.env.unwrapped.task_sampler.families:
            task = self.env.unwrapped.sample_meta_task(family=family)
            self.env.unwrapped.set_meta_task(task)
            family_reward = self._rollout_eval_reward()
            rewards.append(family_reward)
            rewards_by_family[family] = family_reward
            self.writer.add_scalar(f"Meta/EvalReward/{family}", family_reward, self.meta_step)

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        std_reward = float(np.std(rewards)) if rewards else 0.0
        self.writer.add_scalar("Meta/EvalMeanReward", mean_reward, self.meta_step)
        self.writer.add_scalar("Meta/EvalStdReward", std_reward, self.meta_step)
        self.agents.set_running_mode("train")
        return mean_reward

    def _rollout_eval_reward(self) -> float:
        reward_sum = 0.0
        states, _ = self.env.reset()
        for timestep in tqdm.tqdm(range(self.eval_timesteps), disable=self.disable_progressbar, file=sys.stdout):
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.eval_timesteps)[0]
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)
                reward_sum += float(rewards.mean().item())

            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    states, _ = self.env.reset()
                else:
                    states = next_states
        return reward_sum

    def _collect_task_delta(self, task: QuadTask, base_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _inner_update_steps(self, num_updates: int):
        self.agents.set_running_mode("train")
        for _ in range(num_updates):
            self._train_single_rollout()

    def _train_single_rollout(self):
        states, _ = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

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

            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, _ = self.env.reset()
                else:
                    states = next_states

    def _snapshot_params(self) -> dict[str, torch.Tensor]:
        params = {}
        for key, value in self.agents.policy.state_dict().items():
            params[f"policy::{key}"] = value.detach().clone()
        for key, value in self.agents.value.state_dict().items():
            params[f"value::{key}"] = value.detach().clone()
        return params

    @torch.no_grad()
    def _load_params(self, flat_state: dict[str, torch.Tensor]):
        policy_state = {k.split("::", 1)[1]: v for k, v in flat_state.items() if k.startswith("policy::")}
        value_state = {k.split("::", 1)[1]: v for k, v in flat_state.items() if k.startswith("value::")}
        self.agents.policy.load_state_dict(policy_state, strict=True)
        self.agents.value.load_state_dict(value_state, strict=True)

    @torch.no_grad()
    def _add_scaled(self, acc: dict[str, torch.Tensor], delta: dict[str, torch.Tensor], scale: float):
        for key in acc:
            acc[key].add_(scale * delta[key])

    def _delta_norm(self, delta: dict[str, torch.Tensor]) -> float:
        return float(sum((value**2).sum().item() for value in delta.values()) ** 0.5)

    def _delta_to_grad_proxy(self, delta: dict[str, torch.Tensor], step_size: float) -> dict[str, torch.Tensor]:
        del step_size
        # If update is theta <- theta + lr * delta and theta <- theta - lr * g,
        # then gradient proxy is g ~= -delta.
        return {k: -delta[k] for k in delta}

    def _log_grad_proxy_by_module(self, grad_proxy: dict[str, torch.Tensor], tag: str) -> None:
        policy_items = [v for k, v in grad_proxy.items() if k.startswith("policy::")]
        value_items = [v for k, v in grad_proxy.items() if k.startswith("value::")]

        if policy_items:
            policy_norm = float(sum((v**2).sum().item() for v in policy_items) ** 0.5)
            self.writer.add_scalar(f"Meta/GradProxyNorm/{tag}/Policy", policy_norm, self.meta_step)
        if value_items:
            value_norm = float(sum((v**2).sum().item() for v in value_items) ** 0.5)
            self.writer.add_scalar(f"Meta/GradProxyNorm/{tag}/Value", value_norm, self.meta_step)

    def _estimate_avg_task_delta(
        self, task_batch: list[QuadTask], reference_state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        avg_delta = {k: torch.zeros_like(v) for k, v in reference_state.items()}
        for task in task_batch:
            self._load_params(reference_state)
            task_delta = self._collect_task_delta(task, reference_state)
            self._add_scaled(avg_delta, task_delta, 1.0 / len(task_batch))
        return avg_delta


class ReptileMetaTrainer(BaseMetaTrainer):
    """Reptile-style first-order meta trainer."""

    def _collect_task_delta(self, task: QuadTask, base_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.env.unwrapped.set_meta_task(task)
        self._inner_update_steps(self.inner_updates)
        adapted_state = self._snapshot_params()
        return {k: adapted_state[k] - base_state[k] for k in base_state}


class MAMLMetaTrainer(BaseMetaTrainer):
    """First-order MAML approximation.

    Supports distinct support/query adaptation by sampling a query task from the same family.
    """

    def __init__(self, env, agents, agents_scope=None, cfg=None):
        super().__init__(env, agents, agents_scope=agents_scope, cfg=cfg)
        required = ("build_batch_from_rollout", "differentiable_inner_update", "compute_query_loss")
        for name in required:
            if not hasattr(self.agents, name):
                raise TypeError(f"MAMLMetaTrainer requires agent with '{name}' method")
        self._meta_optim = Adam(
            list(self.agents.policy.parameters()) + list(self.agents.value.parameters()), lr=self.meta_lr
        )

    def train(self):
        if self.query_updates < 1:
            raise ValueError("MAMLMetaTrainer requires query_updates >= 1 to compute a query meta signal")

        for _ in range(self.meta_iterations):
            task_batch = self.env.unwrapped.task_sampler.sample_batch(self.tasks_per_meta)
            if self.log_task_batch:
                self.meta_logger.log_task_batch(task_batch, step=self.meta_step + 1, prefix="Meta/MAML/TaskBatch")
            meta_loss = 0.0
            self._meta_optim.zero_grad(set_to_none=True)
            support_policy_losses = []
            support_value_losses = []
            query_policy_losses = []
            query_value_losses = []

            for task in task_batch:
                if self.log_task_params:
                    self.meta_logger.log_task(task, step=self.meta_step + 1, prefix="Meta/MAML/Task")
                support_rollout = self._collect_rollout(task)
                support_batch = self.agents.build_batch_from_rollout(support_rollout)

                fpolicy, fvalue, support_logs = self.agents.differentiable_inner_update(
                    batch=support_batch,
                    inner_steps=self.inner_updates,
                    inner_lr=self.meta_lr,
                    track_higher_grads=True,
                )

                query_task = self.env.unwrapped.task_sampler.sample_like(task)
                query_rollout = self._collect_rollout_with_functional_policy(query_task, fpolicy, fvalue)
                query_batch = self.agents.build_batch_from_rollout(query_rollout)

                query_loss, query_info = self.agents.compute_query_loss(
                    query_batch=query_batch,
                    policy=fpolicy,
                    value=fvalue,
                )

                (query_loss / self.tasks_per_meta).backward()
                meta_loss += query_loss.detach().item()

                if support_logs:
                    support_policy_losses.append(float(support_logs[-1]["policy_loss"]))
                    support_value_losses.append(float(support_logs[-1]["value_loss"]))

                query_policy_losses.append(float(query_info["policy_loss"].item()))
                query_value_losses.append(float(query_info["value_loss"].item()))

            grad_norm = self._grad_norm(self.agents.policy, self.agents.value)
            self.writer.add_scalar("Meta/MAML/GradNorm", grad_norm, self.meta_step)

            self._meta_optim.step()

            self.meta_step += 1
            self.writer.add_scalar("Meta/MAML/MetaLoss", meta_loss / self.tasks_per_meta, self.meta_step)

            if support_policy_losses:
                self.meta_logger.log_scalars(
                    "Meta/MAML/SupportLoss",
                    {
                        "PolicyMean": float(np.mean(support_policy_losses)),
                        "PolicyStd": float(np.std(support_policy_losses)),
                        "ValueMean": float(np.mean(support_value_losses)),
                        "ValueStd": float(np.std(support_value_losses)),
                    },
                    step=self.meta_step,
                )
            if query_policy_losses:
                self.meta_logger.log_scalars(
                    "Meta/MAML/QueryLoss",
                    {
                        "PolicyMean": float(np.mean(query_policy_losses)),
                        "PolicyStd": float(np.std(query_policy_losses)),
                        "ValueMean": float(np.mean(query_value_losses)),
                        "ValueStd": float(np.std(query_value_losses)),
                    },
                    step=self.meta_step,
                )

            if self.eval_interval > 0 and self.meta_step % self.eval_interval == 0:
                self.evaluate_meta()

    def _collect_rollout(self, task: QuadTask):
        self.env.unwrapped.set_meta_task(task)
        return self._collect_rollout_with_policy()

    def _collect_rollout_with_policy(self):
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

        states, _ = self.env.reset()

        for t in range(self.timesteps):
            with torch.no_grad():
                actions, log_prob, _ = self.agents.act(states, timestep=t, timesteps=self.timesteps)
                values, _, _ = self.agents.value.act(
                    {"states": self.agents._state_preprocessor(states)}, role="value"
                )
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)

            rollout["states"].append(states.detach())
            rollout["actions"].append(actions.detach())
            rollout["rewards"].append(rewards.detach())
            rollout["next_states"].append(next_states.detach())
            rollout["terminated"].append(terminated.detach())
            rollout["truncated"].append(truncated.detach())
            rollout["log_prob"].append(log_prob.detach())
            rollout["values"].append(values.detach())

            states = next_states

        for key in rollout:
            rollout[key] = torch.stack(rollout[key], dim=0)

        return rollout

    def _collect_rollout_with_functional_policy(self, task: QuadTask, fpolicy, fvalue):
        self.env.unwrapped.set_meta_task(task)
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

        states, _ = self.env.reset()

        for t in range(self.timesteps):
            with torch.no_grad():
                proc_states = self.agents._state_preprocessor(states)
                actions, log_prob, _ = fpolicy.act({"states": proc_states}, role="policy")
                values, _, _ = fvalue.act({"states": proc_states}, role="value")
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)

            rollout["states"].append(states.detach())
            rollout["actions"].append(actions.detach())
            rollout["rewards"].append(rewards.detach())
            rollout["next_states"].append(next_states.detach())
            rollout["terminated"].append(terminated.detach())
            rollout["truncated"].append(truncated.detach())
            rollout["log_prob"].append(log_prob.detach())
            rollout["values"].append(values.detach())

            states = next_states

        for key in rollout:
            rollout[key] = torch.stack(rollout[key], dim=0)

        return rollout

    def _grad_norm(self, policy, value) -> float:
        params = list(policy.parameters()) + list(value.parameters())
        total = 0.0
        for param in params:
            if param.grad is not None:
                total += float(param.grad.detach().pow(2).sum().item())
        return float(total**0.5)
