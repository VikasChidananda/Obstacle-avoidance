"""Domain randomization quadrotor environment.

Applies physics parameter randomization to improve robustness and sim-to-real transfer.
"""

from __future__ import annotations

import torch

from .quadcontrol_env import QuadcontrolEnv
from .quadcontrol_domain_rand_env_cfg import QuadcontrolDomainRandEnvCfg


class QuadcontrolDomainRandEnv(QuadcontrolEnv):
    """Quadrotor environment with domain randomization for sim-to-real transfer."""

    cfg: QuadcontrolDomainRandEnvCfg

    def __init__(self, cfg: QuadcontrolDomainRandEnvCfg, render_mode: str | None = None, verbose: bool = False, **kwargs):
        """Initialize domain randomization environment."""
        super().__init__(cfg, render_mode, verbose, **kwargs)

        # Store base physics parameters for reference during scaling
        self._base_mass = self._robot.root_physx_view.get_masses().clone().to(self.device)

        # Per-environment randomization factors
        self._mass_scales = torch.ones(self.num_envs, device=self.device)
        self._thrust_scales = torch.ones(self.num_envs, device=self.device)
        self._moment_scales = torch.ones(self.num_envs, device=self.device)
        self._gravity_scale = 1.0
        self._drag_scales = torch.ones(self.num_envs, device=self.device)

        if self.cfg.enable_domain_randomization:
            self._randomize_physics()

    def _randomize_physics(self):
        """Apply domain randomization to physics parameters via scaling factors.
        
        Note: We apply randomization as scaling factors rather than modifying backend state,
        since IsaacLab's root_physx_view does not expose set_masses() or set_inertias().
        The scaling factors are applied during physics step computation.
        """
        if not self.cfg.enable_domain_randomization:
            return

        # Randomize mass scale (simulates mass variability without backend modification)
        mass_min, mass_max = self.cfg.mass_range
        nominal_mass = self._base_mass.sum(dim=-1).to(self.device)
        target_masses = torch.empty(self.num_envs, device=self.device).uniform_(mass_min, mass_max)
        self._mass_scales = target_masses / nominal_mass

        # Randomize per-environment control scales
        thrust_min, thrust_max = self.cfg.thrust_range
        self._thrust_scales = torch.empty(self.num_envs, device=self.device).uniform_(thrust_min, thrust_max)

        moment_min, moment_max = self.cfg.moment_range
        self._moment_scales = torch.empty(self.num_envs, device=self.device).uniform_(moment_min, moment_max)

        # Gravity randomization (globally applied, affects reward computation)
        grav_min, grav_max = self.cfg.gravity_range
        self._gravity_scale = torch.empty(1, device=self.device).uniform_(grav_min, grav_max).item() / 9.81

        # Drag scale (air resistance variability)
        drag_min, drag_max = self.cfg.drag_range
        self._drag_scales = torch.empty(self.num_envs, device=self.device).uniform_(drag_min, drag_max)

        if self.verbose:
            print(f"Domain randomization applied (via scaling factors):")
            print(f"  Mass scales: {self._mass_scales.min().item():.3f} - {self._mass_scales.max().item():.3f}")
            print(f"  Thrust scales: {self._thrust_scales.min().item():.3f} - {self._thrust_scales.max().item():.3f}")
            print(f"  Moment scales: {self._moment_scales.min().item():.3f} - {self._moment_scales.max().item():.3f}")
            print(f"  Gravity scale: {self._gravity_scale:.3f}")
            print(f"  Drag scales: {self._drag_scales.min().item():.3f} - {self._drag_scales.max().item():.3f}")

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply domain-randomized physics to actions."""
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Apply per-environment thrust and moment scales
        thrust = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        thrust = thrust * self._thrust_scales
        thrust = thrust / (self._mass_scales * self._gravity_scale)

        self._thrust[:, 0, 2] = thrust

        moment = self.cfg.moment_scale * self._actions[:, 1:]
        moment = moment * self._moment_scales.unsqueeze(-1)
        moment = moment / self._mass_scales.unsqueeze(-1)
        self._moment[:, 0, :] = moment

        self.post_physics_step()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments with optional per-reset randomization."""
        super()._reset_idx(env_ids)

        # Re-randomize if configured to do so per episode
        if self.cfg.randomize_on_reset and len(env_ids) > 0:
            # Randomize only the resetting environments
            mass_min, mass_max = self.cfg.mass_range
            nominal_mass = self._base_mass.sum(dim=-1).to(self.device)
            target_masses = torch.empty(self.num_envs, device=self.device).uniform_(mass_min, mass_max)
            self._mass_scales[env_ids] = target_masses[env_ids] / nominal_mass[env_ids]

            thrust_min, thrust_max = self.cfg.thrust_range
            self._thrust_scales[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(thrust_min, thrust_max)

            moment_min, moment_max = self.cfg.moment_range
            self._moment_scales[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(moment_min, moment_max)
