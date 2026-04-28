# Meta RL + skrl integration notes

This README documents the architectural changes made for:

1. experimenting with PPO variants without editing skrl's PPO source,
2. reusing skrl-style inner algorithms inside meta-trainers,
3. improving TensorBoard logging for meta-RL runs.

## What changed

### 1) PPO variants moved to pluggable objectives

Changed files:
- `QuadControl/meta_rl/algos/ppo_objectives.py` (new)
- `QuadControl/meta_rl/algos/custom_ppo.py`

Details:
- Added a policy objective registry-like function: `compute_policy_objective(...)`.
- `CustomPPO` now reads from config:
  - `agent.policy_variant`
  - `agent.policy_variant_kwargs`
- Supported variants:
  - `clip` (standard PPO clipped objective)
  - `unclipped`
  - `kl_penalty` (uses `kl_beta` from `policy_variant_kwargs`)
  - `dual_clip` (uses `dual_clip` from `policy_variant_kwargs`)
- The main PPO update loop is still in `CustomPPO` (subclass approach), but objective logic is now separated, which makes variant experiments cleaner and less error-prone.
- KL statistics are now consistently tracked and used for scheduler stepping.

Why this is better:
- You can compare policy objectives by YAML config only.
- You avoid copy/pasting full update loops for every variant.
- Objective-level experiments are isolated from rollout/memory plumbing.

### 2) Meta trainer logging became structured

Changed files:
- `QuadControl/meta_rl/meta_logging.py` (new)
- `QuadControl/meta_rl/meta_trainers.py`

Details:
- Added `MetaLogger` helper to keep trainer code readable.
- Added trainer config flags:
  - `trainer.log_task_batch`
  - `trainer.log_task_params`
- New logs now include:
  - sampled task family counts per meta-iteration,
  - sampled task parameter mean/std,
  - per-family evaluation reward (`Meta/EvalReward/<family>`),
  - evaluation reward mean and std,
  - MAML support/query loss mean and std (policy/value).

Why this is better:
- Better visibility into whether meta-training improvements correlate with task distributions.
- Easier debugging of support-query mismatch and adaptation quality.

### 3) Runner robustness fix

Changed file:
- `QuadControl/utils/custom_runner.py`

Details:
- Fixed fallback call typo:
  - from `super._generate_agent(...)` (invalid)
  - to `super()._generate_agent(...)` (correct)
- Added small alias handling (`customppo`) in component resolution.

Why this matters:
- Prevents crashes when runner falls back to the base skrl path.

### 4) Config surface updated for experiments

Changed files:
- `QuadControl/tasks/direct/quadcontrol/agents/skrl_maml_cfg.yaml`
- `QuadControl/tasks/direct/quadcontrol/agents/skrl_reptile_cfg.yaml`

Details:
- Added in `agent`:
  - `policy_variant: clip`
  - `policy_variant_kwargs: {}`
- Added in `trainer`:
  - `log_task_batch: True`
  - `log_task_params: True`

## How to use it

## PPO variant sweeps

Edit one of the skrl meta configs:

```yaml
agent:
  class: custom
  policy_variant: kl_penalty
  policy_variant_kwargs:
    kl_beta: 0.02
```

Examples:

```yaml
policy_variant: clip
policy_variant_kwargs: {}
```

```yaml
policy_variant: dual_clip
policy_variant_kwargs:
  dual_clip: 3.0
```

## Meta logging controls

```yaml
trainer:
  log_task_batch: True
  log_task_params: True
```

Disable high-frequency task-parameter logs if disk usage is high:

```yaml
trainer:
  log_task_batch: True
  log_task_params: False
```

## Recommended architecture going forward

For your goals, this pattern works well:

- Keep `CustomPPO` only for skrl-compatible behavior extension points.
- Keep each PPO experiment as an objective function (not as a full algorithm fork).
- Let `MAMLMetaTrainer` / `ReptileMetaTrainer` depend on a minimal inner-agent interface:
  - `build_batch_from_rollout`
  - `differentiable_inner_update`
  - `compute_query_loss`
- Keep meta-level logging in a dedicated helper (`MetaLogger`) so trainer logic stays focused on optimization.

This setup gives you fast iteration on PPO ideas while preserving skrl internals and reducing merge pain when skrl updates.
