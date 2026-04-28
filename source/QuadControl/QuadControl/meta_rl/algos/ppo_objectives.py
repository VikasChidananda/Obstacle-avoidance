from __future__ import annotations

import torch


def compute_policy_objective(
    *,
    variant: str,
    old_log_prob: torch.Tensor,
    new_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    ratio_clip: float,
    variant_kwargs: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute PPO policy loss variants from the same rollout tensors.

    Supported variants:
      - "clip": standard clipped PPO objective
      - "unclipped": unclipped policy gradient surrogate
      - "kl_penalty": unclipped surrogate with forward-KL penalty term
      - "dual_clip": PPO dual-clip objective (useful when advantages are negative)
    """
    opts = variant_kwargs or {}
    ratio = torch.exp(new_log_prob - old_log_prob)
    surrogate = advantages * ratio
    surrogate_clipped = advantages * torch.clamp(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)

    approx_kl_tensor = (old_log_prob - new_log_prob).mean()
    clip_fraction_tensor = (torch.abs(ratio - 1.0) > ratio_clip).float().mean()

    normalized_variant = (variant or "clip").lower()
    if normalized_variant == "clip":
        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
    elif normalized_variant == "unclipped":
        policy_loss = -surrogate.mean()
    elif normalized_variant == "kl_penalty":
        kl_beta = float(opts.get("kl_beta", 0.01))
        policy_loss = -surrogate.mean() + kl_beta * approx_kl_tensor
    elif normalized_variant == "dual_clip":
        dual_clip = float(opts.get("dual_clip", 3.0))
        clipped_objective = torch.min(surrogate, surrogate_clipped)
        dual_clip_objective = torch.max(clipped_objective, dual_clip * advantages)
        policy_objective = torch.where(advantages < 0.0, dual_clip_objective, clipped_objective)
        policy_loss = -policy_objective.mean()
    else:
        raise ValueError(
            f"Unsupported PPO variant '{variant}'. Supported variants: clip, unclipped, kl_penalty, dual_clip"
        )

    stats = {
        "approx_kl": float(approx_kl_tensor.detach().item()),
        "clip_fraction": float(clip_fraction_tensor.detach().item()),
    }
    return policy_loss, stats
