import copy
import itertools
from typing import Any, Optional, Tuple, Union
from collections.abc import Mapping

import gymnasium
import higher  # for differentiable optimization in the inner loop
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

# This ppo has to create an update loop that
#  preserves the computation graph for the inner loop updates, so that we can backprop through it in the outer loop.


class CustomPPO(PPO):
    def __init__(
        self,
        models,
        memory,
        observation_space,
        action_space,
        device,
        cfg,
    ):
        self._update_mode = cfg.get("update_mode", "maml")  # "differentiable" or "standard"
        super().__init__(models, memory, observation_space, action_space, device, cfg)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        discount_factor: float = 0.99,
        lambda_coefficient: float = 0.95,
    ) -> torch.Tensor:
        """Compute the Generalized Advantage Estimator (GAE)

        :param rewards: Rewards obtained by the agent
        :type rewards: torch.Tensor
        :param dones: Signals to indicate that episodes have ended
        :type dones: torch.Tensor
        :param values: Values obtained by the agent
        :type values: torch.Tensor
        :param next_values: Next values obtained by the agent
        :type next_values: torch.Tensor
        :param discount_factor: Discount factor
        :type discount_factor: float
        :param lambda_coefficient: Lambda coefficient
        :type lambda_coefficient: float

        :return: Generalized Advantage Estimator
        :rtype: torch.Tensor
        """
        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_dones = dones.logical_not()
        memory_size = rewards.shape[0]

        # advantages computation
        for i in reversed(range(memory_size)):
            next_values = values[i + 1] if i < memory_size - 1 else values[i]
            advantage = (
                rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
            )
            advantages[i] = advantage
        # returns computation
        returns = advantages + values
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    # updating the PPO update loop to preserve the computation graph for inner loop updates
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        :param differentiable: Whether to preserve the computation graph for differentiable updates
        :type differentiable: bool
        """
        print("Updating PPO...")

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")

        returns, advantages = self.compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # define and outer loss function here that preserves the computation graph for the inner loop updates, so that we can backprop through it in the outer loop.

                    # if not differentiable:
                    policy_loss, value_loss, entropy_loss = self._compute_loss(
                        sampled_states,
                        sampled_actions,
                        sampled_log_prob,
                        sampled_values,
                        sampled_returns,
                        sampled_advantages,
                        policy=self.policy,
                        value=self.value,
                        kl_divergences=kl_divergences,
                        epoch=epoch,
                    )

                    # elif differentiable:
                    #     # for maml, we need to perform inner loop updates on the meta-batch of tasks, and then compute the loss on the query set using the adapted parameters.
                    #     policy_loss, value_loss, entropy_loss = self.differentiable_inner_update(
                    #         batch={
                    #             "sampled_states": sampled_states,
                    #             "sampled_actions": sampled_actions,
                    #             "sampled_log_prob": sampled_log_prob,
                    #             "sampled_values": sampled_values,
                    #             "sampled_returns": sampled_returns,
                    #             "sampled_advantages": sampled_advantages,
                    #         },
                    #         inner_steps=self.inner_loops,
                    #         inner_lr=self.meta_lr,
                    #         track_higher_grads=True
                    #     )[2][-1].values()

                    # for maml
                    # policy_loss, value_loss, entropy_loss = self.differentiable_inner_update(
                    #     meta_batch=[{
                    #         "sampled_states": sampled_states,
                    #         "sampled_actions": sampled_actions,
                    #         "sampled_log_prob": sampled_log_prob,
                    #         "sampled_values": sampled_values,
                    #         "sampled_returns": sampled_returns,
                    #         "sampled_advantages": sampled_advantages,
                    #     }],
                    #     inner_steps=self.inner_loops,
                    #     inner_lr=self.meta_lr,
                    #     track_higher_grads=True
                    # )

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        # return fpolicy, fvalue, {}

    def _compute_loss(
        self,
        sampled_states,
        sampled_actions,
        sampled_log_prob,
        sampled_values,
        sampled_returns,
        sampled_advantages,
        policy,
        value,
        kl_divergences,
        epoch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the PPO loss"""
        sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

        _, next_log_prob, _ = policy.act({"states": sampled_states, "taken_actions": sampled_actions}, role="policy")

        # compute entropy loss
        if self._entropy_loss_scale:
            entropy_loss = -self._entropy_loss_scale * policy.get_entropy(role="policy").mean()
        else:
            entropy_loss = 0

        # compute policy loss
        ratio = torch.exp(next_log_prob - sampled_log_prob)
        surrogate = sampled_advantages * ratio
        surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)

        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

        # compute value loss
        predicted_values, _, _ = value.act({"states": sampled_states}, role="value")

        if self._clip_predicted_values:
            predicted_values = sampled_values + torch.clip(
                predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
            )
        value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

        return policy_loss, value_loss, entropy_loss

    def build_batch_from_rollout(self, rollout):
        with torch.no_grad():
            last_states = rollout["next_states"][-1]
            last_values, _, _ = self.value.act({"states": self._state_preprocessor(last_states.float())}, role="value")
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = rollout["values"]

        returns, advantages = self.compute_gae(
            rewards=rollout["rewards"],
            dones=rollout["terminated"] | rollout["truncated"],
            values=values,
            next_values=last_values,
        )

        batch = {
            "sampled_states": rollout["states"].reshape(-1, *rollout["states"].shape[2:]),
            "sampled_actions": rollout["actions"].reshape(-1, *rollout["actions"].shape[2:]),
            "sampled_log_prob": rollout["log_prob"].reshape(-1, *rollout["log_prob"].shape[2:]),
            "sampled_values": values.reshape(-1, *values.shape[2:]),
            "sampled_returns": returns.reshape(-1, *returns.shape[2:]),
            "sampled_advantages": advantages.reshape(-1, *advantages.shape[2:]),
        }
        return batch

    def differentiable_inner_update(self, batch, inner_steps, inner_lr, track_higher_grads: bool = True):
        """Perform differentiable inner loop updates on the meta-batch of tasks.

        Args:
            meta_batch: A batch of tasks, where each task contains its own data for inner loop updates.
            inner_steps: Number of inner loop update steps to perform.
            inner_lr: Learning rate for the inner loop updates.
            track_higher_grads: Whether to track higher-order gradients for meta-optimization.
        Returns:
            A list of adapted parameters for each task in the meta-batch after performing inner loop updates
        """

        inner_lr = inner_lr if isinstance(inner_lr, float) else self._learning_rate

        # optimizers for both policy and value networks
        policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=inner_lr)
        value_optimizer = torch.optim.Adam(self.value.parameters(), lr=inner_lr)

        logs = []

        # higher context for differentiable optimization
        with higher.innerloop_ctx(
            self.policy,
            policy_optimizer,
            copy_initial_weights=False,
            track_higher_grads=track_higher_grads,
        ) as (fpolicy, diff_policy_opt), higher.innerloop_ctx(
            self.value,
            value_optimizer,
            copy_initial_weights=False,
            track_higher_grads=track_higher_grads,
        ) as (
            fvalue,
            diff_value_opt,
        ):

            # perform inner loop updates for the current task
            for _ in range(inner_steps):
                policy_loss, value_loss, entropy_loss = self._compute_loss(
                    batch["sampled_states"],
                    batch["sampled_actions"],
                    batch["sampled_log_prob"],
                    batch["sampled_values"],
                    batch["sampled_returns"],
                    batch["sampled_advantages"],
                    policy=fpolicy,
                    value=fvalue,
                    kl_divergences=[],
                    epoch=0,
                )

                # entropy_loss =

                inner_loss = policy_loss + value_loss + entropy_loss

                diff_policy_opt.step(inner_loss)
                diff_value_opt.step(inner_loss)

                logs.append({
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy_loss": entropy_loss,
                })

        return fpolicy, fvalue, logs

    def compute_query_loss(self, query_batch, policy, value):

        policy_loss, value_loss, entropy_loss = self._compute_loss(
            query_batch["sampled_states"],
            query_batch["sampled_actions"],
            query_batch["sampled_log_prob"],
            query_batch["sampled_values"],
            query_batch["sampled_returns"],
            query_batch["sampled_advantages"],
            policy=policy,
            value=value,
            kl_divergences=[],
            epoch=0,
        )
        total_loss = policy_loss + value_loss + entropy_loss

        return total_loss, {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy_loss": entropy_loss,
        }

    # updating the PPO update loop to preserve the computation graph for inner loop updates
    def _differentiable_update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        :param differentiable: Whether to preserve the computation graph for differentiable updates
        :type differentiable: bool
        """
        print("Updating PPO...")

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")

        returns, advantages = self.compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # define and outer loss function here that preserves the computation graph for the inner loop updates, so that we can backprop through it in the outer loop.

                    # if not differentiable:
                    policy_loss, value_loss, entropy_loss = self._compute_loss(
                        sampled_states,
                        sampled_actions,
                        sampled_log_prob,
                        sampled_values,
                        sampled_returns,
                        sampled_advantages,
                        policy=self.policy,
                        value=self.value,
                        kl_divergences=kl_divergences,
                        epoch=epoch,
                    )
                    # elif differentiable:
                    #     # for maml, we need to perform inner loop updates on the meta-batch of tasks, and then compute the loss on the query set using the adapted parameters.
                    #     policy_loss, value_loss, entropy_loss = self.differentiable_inner_update(
                    #         batch={
                    #             "sampled_states": sampled_states,
                    #             "sampled_actions": sampled_actions,
                    #             "sampled_log_prob": sampled_log_prob,
                    #             "sampled_values": sampled_values,
                    #             "sampled_returns": sampled_returns,
                    #             "sampled_advantages": sampled_advantages,
                    #         },
                    #         inner_steps=self.inner_loops,
                    #         inner_lr=self.meta_lr,
                    #         track_higher_grads=True
                    #     )[2][-1].values()

                    # for maml
                    # policy_loss, value_loss, entropy_loss = self.differentiable_inner_update(
                    #     meta_batch=[{
                    #         "sampled_states": sampled_states,
                    #         "sampled_actions": sampled_actions,
                    #         "sampled_log_prob": sampled_log_prob,
                    #         "sampled_values": sampled_values,
                    #         "sampled_returns": sampled_returns,
                    #         "sampled_advantages": sampled_advantages,
                    #     }],
                    #     inner_steps=self.inner_loops,
                    #     inner_lr=self.meta_lr,
                    #     track_higher_grads=True
                    # )

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
