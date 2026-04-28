"""Domain randomization configuration for quadrotor learning.

This config extends QuadcontrolEnvCfg with physics parameter randomization
to improve sim-to-real transfer and robustness, inspired by:
"Learning to Fly in Seconds" - Kaufmann et al., 2021
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .quadcontrol_env_cfg import QuadcontrolEnvCfg


@configclass
class QuadcontrolDomainRandEnvCfg(QuadcontrolEnvCfg):
    """Domain randomized quadrotor environment config."""

    # Domain randomization ranges for physics parameters
    # (format: [min, max] as multipliers or absolute ranges)

    # Mass randomization: [min_mass, max_mass] kg
    mass_range: tuple[float, float] = (0.025, 0.042)  # Crazyflie is ~35g nominal

    # Inertia randomization: [min_scale, max_scale] multiplier
    inertia_range: tuple[float, float] = (0.8, 1.2)

    # Thrust randomization: [min_scale, max_scale] - motor response variability
    thrust_range: tuple[float, float] = (0.9, 1.1)

    # Moment (torque) randomization: [min_scale, max_scale]
    moment_range: tuple[float, float] = (0.9, 1.1)

    # Gravity randomization: [min_g, max_g] m/s^2
    gravity_range: tuple[float, float] = (8.5, 10.5)

    # Drag coefficient randomization: [min_scale, max_scale]
    # Simulates air resistance variability due to wind, altitude, etc.
    drag_range: tuple[float, float] = (0.8, 1.2)

    # Linear damping (air resistance) per axis: [min, max]
    lin_damping_range: tuple[float, float] = (0.0, 0.1)

    # Angular damping per axis: [min, max]
    ang_damping_range: tuple[float, float] = (0.0, 0.05)

    # Flag to enable/disable domain randomization
    enable_domain_randomization: bool = True

    # Randomize on reset or only at initialization
    randomize_on_reset: bool = False  # True = re-randomize per episode, False = once at startup
