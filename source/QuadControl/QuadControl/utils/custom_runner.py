import copy
import logging
from typing import Any, Type, Union
from collections.abc import Mapping

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.models.torch import Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise  # noqa
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa
from skrl.trainers.torch import Trainer
from skrl.utils import set_seed
from skrl.utils.runner.torch.runner import Runner


class CustomRunner(Runner):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        logger.info("Initialized CustomRunner with config:")
        # logger.info(cfg)

    def _component(self, name: str):  # type: ignore[override]
        # Map meta_ppo identifiers to PPO and its default config
        lname = name.lower()
        if lname in ("ppo", "custom"):
            # Reuse PPO components from skrl
            from QuadControl.meta_rl.algos.custom_ppo import CustomPPO

            print("Using CustomPPO")
            return CustomPPO

        elif lname in ("spo"):
            from QuadControl.meta_rl.algos.spo import SPO

            print("Using SPO")
            return SPO

        # trainers
        elif lname in ("metatrainer"):
            # Reuse PPO components from skrl
            from QuadControl.meta_rl.meta_trainer_maml import MetaTrainer

            print("Using MetaTrainer")
            return MetaTrainer
        # fallback to the base resolution
        return super()._component(name)

    # def _generate_agent(
    #     self,
    #     env,
    #     cfg,
    #     models,
    # ):
    #     # If the requested agent is meta_ppo, transparently treat it as ppo for generation
    #     try:
    #         agent_class = cfg.get("agent", {}).get("class", "")
    #         print(f"CustomRunner agent class: {agent_class}")
    #     except Exception:
    #         agent_class = ""
    #     if isinstance(agent_class, str) and agent_class.lower() == "custom":
    #         cfg = copy.deepcopy(cfg)
    #         # Preserve original for logging but use PPO under the hood
    #         cfg["agent"]["class"] = "custom"
    #         print(f"CustomRunner using PPO")
    #     if isinstance(agent_class, str) and agent_class.lower() == "spo":
    #         cfg = copy.deepcopy(cfg)
    #         # Preserve original for logging but use SPO under the hood
    #         cfg["agent"]["class"] = "spo"
    #         print(f"CustomRunner using SPO")
    #     return super()._generate_agent(env, cfg, models)

    def _generate_agent(
        self,
        env: Wrapper | MultiAgentEnvWrapper,
        cfg: Mapping[str, Any],
        models: Mapping[str, Mapping[str, Model]],
    ) -> Agent:
        """Generate agent instance according to the environment specification and the given config and models

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param models: Agent's model instances

        :return: Agent instances
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        num_envs = env.num_envs
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        print(f"-------------------------------------cfg : {cfg}")
        agent_class = cfg.get("agent", {}).get("class", "").lower()

        # for custom agent handlinig of specific configs for my implementation
        # -------------------------------------------------------------------------------------------------------
        if isinstance(agent_class, str) and agent_class.lower() == "custom":
            cfg = copy.deepcopy(cfg)
            # Preserve original for logging but use PPO under the hood
            cfg["agent"]["class"] = "custom"
            print(f"CustomRunner using PPO")
        if isinstance(agent_class, str) and agent_class.lower() == "spo":
            cfg = copy.deepcopy(cfg)
            # Preserve original for logging but use SPO under the hood
            cfg["agent"]["class"] = "spo"
            print(f"CustomRunner using SPO")

        if not agent_class:
            raise ValueError(f"No 'class' field defined in 'agent' cfg")
        # -------------------------------------------------------------------------------------------------------

        # check for memory configuration (backward compatibility)
        if not "memory" in cfg:
            logger.warning(
                "Deprecation warning: No 'memory' field defined in cfg. Using the default generated configuration"
            )
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        # get memory class and remove 'class' field
        try:
            memory_class = self._component(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._component("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        memories = {}
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # Taken care of by the source implementation of _generate_agent in Runner, but left here for reference and potential future use

        # # single-agent configuration and instantiation
        # if agent_class in ["amp"]:
        #     agent_id = possible_agents[0]
        #     try:
        #         amp_observation_space = env.amp_observation_space
        #     except Exception as e:
        #         logger.warning(
        #             "Unable to get AMP space via 'env.amp_observation_space'. Using 'env.observation_space' instead"
        #         )
        #         amp_observation_space = observation_spaces[agent_id]
        #     agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
        #     agent_cfg.update(self._process_cfg(cfg["agent"]))
        #     agent_cfg["state_preprocessor_kwargs"].update({"size": observation_spaces[agent_id], "device": device})
        #     agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
        #     agent_cfg["amp_state_preprocessor_kwargs"].update({"size": amp_observation_space, "device": device})

        #     motion_dataset = None
        #     if cfg.get("motion_dataset"):
        #         motion_dataset_class = cfg["motion_dataset"].get("class")
        #         if not motion_dataset_class:
        #             raise ValueError(f"No 'class' field defined in 'motion_dataset' cfg")
        #         del cfg["motion_dataset"]["class"]
        #         motion_dataset = self._component(motion_dataset_class)(
        #             device=device, **self._process_cfg(cfg.get("motion_dataset", {}))
        #         )
        #     reply_buffer = None
        #     if cfg.get("reply_buffer"):
        #         reply_buffer_class = cfg["reply_buffer"].get("class")
        #         if not reply_buffer_class:
        #             raise ValueError(f"No 'class' field defined in 'reply_buffer' cfg")
        #         del cfg["reply_buffer"]["class"]
        #         reply_buffer = self._component(reply_buffer_class)(
        #             device=device, **self._process_cfg(cfg.get("reply_buffer", {}))
        #         )

        #     agent_kwargs = {
        #         "models": models[agent_id],
        #         "memory": memories[agent_id],
        #         "observation_space": observation_spaces[agent_id],
        #         "action_space": action_spaces[agent_id],
        #         "amp_observation_space": amp_observation_space,
        #         "motion_dataset": motion_dataset,
        #         "reply_buffer": reply_buffer,
        #         "collect_reference_motions": lambda num_samples: env.collect_reference_motions(num_samples),
        #     }

        if agent_class in ["custom", "spo"]:
            agent_id = possible_agents[0]
            agent_cfg = self._component(f"PPO_DEFAULT_CONFIG").copy()
            agent_cfg = self._process_cfg(cfg["agent"])
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})

            if agent_cfg.get("exploration", {}).get("noise", None):
                agent_cfg["exploration"].get("noise_kwargs", {}).update({"device": device})
                agent_cfg["exploration"]["noise"] = agent_cfg["exploration"]["noise"](
                    **agent_cfg["exploration"].get("noise_kwargs", {})
                )
            if agent_cfg.get("smooth_regularization_noise", None):
                agent_cfg.get("smooth_regularization_noise_kwargs", {}).update({"device": device})
                agent_cfg["smooth_regularization_noise"] = agent_cfg["smooth_regularization_noise"](
                    **agent_cfg.get("smooth_regularization_noise_kwargs", {})
                )

            print(f"---------------------------------------------------------CustomRunner agent cfg: {agent_cfg}")

            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
            return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)

        # --- Taken care by the source implmentation of _generate_agent in Runner, but left here for reference and potential future use ---

        # elif agent_class in ["a2c", "cem", "ddpg", "ddqn", "dqn", "ppo", "rpo", "sac", "td3", "trpo"]:
        #     agent_id = possible_agents[0]
        #     agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
        #     agent_cfg.update(self._process_cfg(cfg["agent"]))
        #     agent_cfg.get("state_preprocessor_kwargs", {}).update(
        #         {"size": observation_spaces[agent_id], "device": device}
        #     )
        #     agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
        #     if agent_cfg.get("exploration", {}).get("noise", None):
        #         agent_cfg["exploration"].get("noise_kwargs", {}).update({"device": device})
        #         agent_cfg["exploration"]["noise"] = agent_cfg["exploration"]["noise"](
        #             **agent_cfg["exploration"].get("noise_kwargs", {})
        #         )
        #     if agent_cfg.get("smooth_regularization_noise", None):
        #         agent_cfg.get("smooth_regularization_noise_kwargs", {}).update({"device": device})
        #         agent_cfg["smooth_regularization_noise"] = agent_cfg["smooth_regularization_noise"](
        #             **agent_cfg.get("smooth_regularization_noise_kwargs", {})
        #         )
        #     agent_kwargs = {
        #         "models": models[agent_id],
        #         "memory": memories[agent_id],
        #         "observation_space": observation_spaces[agent_id],
        #         "action_space": action_spaces[agent_id],
        #     }
        # # multi-agent configuration and instantiation
        # elif agent_class in ["ippo"]:
        #     agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
        #     agent_cfg.update(self._process_cfg(cfg["agent"]))
        #     agent_cfg["state_preprocessor_kwargs"].update(
        #         {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
        #     )
        #     agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
        #     agent_kwargs = {
        #         "models": models,
        #         "memories": memories,
        #         "observation_spaces": observation_spaces,
        #         "action_spaces": action_spaces,
        #         "possible_agents": possible_agents,
        #     }
        # elif agent_class in ["mappo"]:
        #     agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
        #     agent_cfg.update(self._process_cfg(cfg["agent"]))
        #     agent_cfg["state_preprocessor_kwargs"].update(
        #         {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
        #     )
        #     agent_cfg["shared_state_preprocessor_kwargs"].update(
        #         {agent_id: {"size": state_spaces[agent_id], "device": device} for agent_id in possible_agents}
        #     )
        #     agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})

        #     agent_kwargs = {
        #         "models": models,
        #         "memories": memories,
        #         "observation_spaces": observation_spaces,
        #         "action_spaces": action_spaces,
        #         "shared_observation_spaces": state_spaces,
        #         "possible_agents": possible_agents,
        #     }
        #     print(f"Agent kwargs: {agent_kwargs}")
        #     print(f"Agent cfg: {agent_cfg}")

        return super._generate_agent(env, cfg, models)

    def _generate_trainer(self, env, cfg, agent):
        """Generate trainer instance according to the environment specification and the given config and agent

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param agent: Agent's model instances

        :return: Trainer instances
        """
        # get trainer class and remove 'class' field
        try:
            trainer_class = self._component(cfg["trainer"]["class"])
            del cfg["trainer"]["class"]
        except KeyError:
            trainer_class = self._component("SequentialTrainer")
            logger.warning("No 'class' field defined in 'trainer' cfg. 'SequentialTrainer' will be used as default")
        # instantiate trainer
        return trainer_class(env=env, agents=agent, cfg=cfg["trainer"])
