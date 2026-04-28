# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Quadcontrol-v0",
    entry_point=f"{__name__}.quadcontrol_env:QuadcontrolEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcontrol_env_cfg:QuadcontrolEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_spo_cfg_entry_point": f"{agents.__name__}:skrl_spo_cfg.yaml",
        "skrl_ppo-rnn_cfg_entry_point": f"{agents.__name__}:skrl_ppo-rnn_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg_default.yaml",
        "skrl_custom_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_custom.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)


gym.register(
    id="Quadcontrol-SymmAug-v0",
    entry_point=f"{__name__}.quadcontrol_env:QuadcontrolEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcontrol_env_cfg:QuadcontrolEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOSymmAugRunnerCfg",
    },
)


gym.register(
    id="Quadcontrol-SymmAugC4-v0",
    entry_point=f"{__name__}.quadcontrol_env:QuadcontrolEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcontrol_env_cfg:QuadcontrolEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOSymmAugC4RunnerCfg",
    },
)


gym.register(
    id="Quadcontrol-SymmAugTheta-v0",
    entry_point=f"{__name__}.quadcontrol_env:QuadcontrolEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcontrol_env_cfg:QuadcontrolEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOSymmAugThetaRunnerCfg",
    },
)


gym.register(
    id="Quadcontrol-SymmAugContinuous-v0",
    entry_point=f"{__name__}.quadcontrol_env:QuadcontrolEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcontrol_env_cfg:QuadcontrolEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOSymmAugContinuousRunnerCfg",
    },
)


gym.register(
    id="Quadcontrol-MirrorLoss-v0",
    entry_point=f"{__name__}.quadcontrol_env:QuadcontrolEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcontrol_env_cfg:QuadcontrolEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOMirrorLossRunnerCfg",
    },
)


gym.register(
    id="Quadcontrol-SymmAugMirror-v0",
    entry_point=f"{__name__}.quadcontrol_env:QuadcontrolEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcontrol_env_cfg:QuadcontrolEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOSymmAugMirrorLossRunnerCfg",
    },
)

gym.register(
    id="Quadcontrol-Meta-v0",
    entry_point=f"{__name__}.quadcontrol_meta_env:QuadcontrolMetaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcontrol_meta_env_cfg:QuadcontrolMetaEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_maml_cfg.yaml",
        "skrl_maml_cfg_entry_point": f"{agents.__name__}:skrl_maml_cfg.yaml",
        "skrl_reptile_cfg_entry_point": f"{agents.__name__}:skrl_reptile_cfg.yaml",
    },
)


gym.register(
    id="Quadcontrol-MetaMAML-v0",
    entry_point=f"{__name__}.quadcontrol_meta_env:QuadcontrolMetaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcontrol_meta_env_cfg:QuadcontrolMetaEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_maml_cfg.yaml",
        "skrl_maml_cfg_entry_point": f"{agents.__name__}:skrl_maml_cfg.yaml",
    },
)


gym.register(
    id="Quadcontrol-MetaReptile-v0",
    entry_point=f"{__name__}.quadcontrol_meta_env:QuadcontrolMetaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcontrol_meta_env_cfg:QuadcontrolMetaEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_reptile_cfg.yaml",
        "skrl_reptile_cfg_entry_point": f"{agents.__name__}:skrl_reptile_cfg.yaml",
    },
)
