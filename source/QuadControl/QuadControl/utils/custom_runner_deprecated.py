import copy
import logging

from skrl import logger
from skrl.utils.runner.torch.runner import Runner


class CustomRunner(Runner):

    def _component(self, name: str):  # type: ignore[override]
        # Map meta_ppo identifiers to PPO and its default config
        lname = name.lower()
        if lname in ("custom"):
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
            from QuadControl.utils.meta_trainer import MetaTrainer

            print("Using MetaTrainer")
            return MetaTrainer
        # fallback to the base resolution
        return super()._component(name)

    def _generate_agent(
        self,
        env,
        cfg,
        models,
    ):
        # If the requested agent is meta_ppo, transparently treat it as ppo for generation
        try:
            agent_class = cfg.get("agent", {}).get("class", "")
            print(f"CustomRunner agent class: {agent_class}")
        except Exception:
            agent_class = ""
        if isinstance(agent_class, str) and agent_class.lower() == "custom":
            cfg = copy.deepcopy(cfg)
            # Preserve original for logging but use PPO under the hood
            cfg["agent"]["class"] = "ppo"
            print(f"CustomRunner using PPO")
        elif isinstance(agent_class, str) and agent_class.lower() == "spo":
            cfg = copy.deepcopy(cfg)
            # Preserve original for logging but use SPO under the hood
            cfg["agent"]["class"] = "spo"
            print(f"CustomRunner using SPO")
        return super()._generate_agent(env, cfg, models)

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
