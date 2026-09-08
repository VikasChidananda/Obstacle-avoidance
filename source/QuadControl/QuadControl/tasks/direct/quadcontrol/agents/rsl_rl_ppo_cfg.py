# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..symmetry import (
    yaw_equivariant_augmentation,
    yaw_equivariant_augmentation_c4,
    yaw_equivariant_augmentation_discretized,
    yaw_equivariant_augmentation_continuous,
)


def _base_ppo_algorithm_cfg() -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "symmetry/baselines"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = _base_ppo_algorithm_cfg()


@configclass
class PPOSymmAugRunnerCfg(PPORunnerCfg):
    experiment_name = "symmetry/aug_only"
    algorithm = _base_ppo_algorithm_cfg().replace(
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=yaw_equivariant_augmentation,
            mirror_loss_coeff=0.0,
        )
    )


@configclass
class PPOSymmAugC4RunnerCfg(PPORunnerCfg):
    experiment_name = "symmetry/aug_only_c4"
    algorithm = _base_ppo_algorithm_cfg().replace(
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=yaw_equivariant_augmentation_c4,
            mirror_loss_coeff=0.0,
        )
    )


@configclass
class PPOSymmAugThetaRunnerCfg(PPORunnerCfg):
    experiment_name = "symmetry/aug_only_theta"
    algorithm = _base_ppo_algorithm_cfg().replace(
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=yaw_equivariant_augmentation_discretized,
            mirror_loss_coeff=0.0,
        )
    )


@configclass
class PPOSymmAugContinuousRunnerCfg(PPORunnerCfg):
    experiment_name = "symmetry/aug_only_continuous"
    algorithm = _base_ppo_algorithm_cfg().replace(
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=yaw_equivariant_augmentation_continuous,
            mirror_loss_coeff=0.0,
        )
    )


@configclass
class PPOMirrorLossRunnerCfg(PPORunnerCfg):
    experiment_name = "symmetry/mirror_loss_only"
    algorithm = _base_ppo_algorithm_cfg().replace(
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=yaw_equivariant_augmentation,
            mirror_loss_coeff=1.0,
        )
    )


@configclass
class PPOSymmAugMirrorLossRunnerCfg(PPORunnerCfg):
    experiment_name = "symmetry/aug_plus_mirror_loss"
    algorithm = _base_ppo_algorithm_cfg().replace(
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=yaw_equivariant_augmentation,
            mirror_loss_coeff=0.2,
        )
    )


@configclass
class PPOQuadrantSymmAugMirrorRunnerCfg(PPORunnerCfg):
    """Train on 1st-quadrant goals with C4 augmentation + mirror loss.

    Uses the full C4 yaw group (0/90/180/270 deg) as data augmentation so the
    policy sees the 1st quadrant mapped onto every other quadrant, and adds a
    mirror loss to enforce equivariant action predictions.
    """

    experiment_name = "symmetry/quadrant_generalize"
    algorithm = _base_ppo_algorithm_cfg().replace(
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=yaw_equivariant_augmentation_c4,
            mirror_loss_coeff=0.2,
        )
    )


@configclass
class PPOFull3DSymmAugMirrorRunnerCfg(PPOQuadrantSymmAugMirrorRunnerCfg):
    """Evaluation config for the quadrant-generalisation experiment.

    Shares ``experiment_name`` with the training config so the eval script can
    locate the checkpoint produced by ``Quadcontrol-QuadrantGoal-SymmAugMirror-v0``.
    """

    experiment_name = "symmetry/quadrant_generalize"


@configclass
class PPOSymmRunnerCfg(PPOSymmAugRunnerCfg):
    """Backward-compatible alias for older experiment scripts."""
