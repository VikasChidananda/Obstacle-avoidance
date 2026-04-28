# Quadrotor Meta-RL (MAML/Reptile) Guide

## What was added

### 1) Task list + sampler
A dedicated task sampler was added in `QuadControl/meta_rl/task_sampler.py`.

Supported task families:
- `hover`: fixed 3D point tracking
- `straight`: linear tracking with sampled direction and speed
- `circle`: planar circular trajectory with optional vertical modulation
- `lemniscate`: figure-eight style trajectory

Each sampled task is represented as:
- `family`: task family name
- `params`: sampled scalar parameters for that family

### 2) Meta trainer implementations (skrl)
Implemented in `QuadControl/meta_rl/meta_trainers.py`:
- `MAMLMetaTrainer`
- `ReptileMetaTrainer`

Both trainers:
- use skrl-compatible sequential rollout/update loops
- sample task batches from the environment task sampler
- apply an outer-loop update on policy/value weights
- periodically evaluate across all configured task families
- write metrics to TensorBoard in the experiment log directory

Algorithm behavior:
- `ReptileMetaTrainer`: adapts on support task and moves base params toward adapted params.
- `MAMLMetaTrainer` (first-order approximation): adapts on support task, then adapts on a query task sampled from the same family before the outer update.

### 3) Isolated subclass environment (base env preserved)
The original symmetry environment remains unchanged.

A meta-only subclass was added:
- `QuadControl/tasks/direct/quadcontrol/quadcontrol_meta_env.py`
- config in `quadcontrol_meta_env_cfg.py`

This subclass:
- keeps original dynamics/reward code path by inheriting from `QuadcontrolEnv`
- adds task-sampled desired-position generation per step
- supports externally forcing a task (`set_meta_task`) for meta inner/outer loops

This ensures your existing symmetry work stays isolated and unaffected.

### 4) Gym registrations and configs
Updated registrations in `QuadControl/tasks/direct/quadcontrol/__init__.py`:
- `Quadcontrol-Meta-v0` (backward-compatible alias)
- `Quadcontrol-MetaMAML-v0`
- `Quadcontrol-MetaReptile-v0`

Added skrl config files:
- `agents/skrl_maml_cfg.yaml`
- `agents/skrl_reptile_cfg.yaml`

### 5) Lightweight train scripts
Added:
- `scripts/skrl/train_meta.py` (single driver with `--meta_algo`)
- `scripts/skrl/train_meta_maml.sh`
- `scripts/skrl/train_meta_reptile.sh`

Examples:

```bash
python scripts/skrl/train_meta.py --meta_algo maml --num_envs 512 --meta_iterations 80
python scripts/skrl/train_meta.py --meta_algo reptile --num_envs 512 --tasks_per_meta 6 --inner_updates 2
```

Or wrappers:

```bash
bash scripts/skrl/train_meta_maml.sh --num_envs 512 --meta_iterations 80
bash scripts/skrl/train_meta_reptile.sh --num_envs 512 --meta_lr 0.08
```

## Recommended tuning knobs

Most useful CLI arguments in `train_meta.py`:
- `--meta_iterations`
- `--tasks_per_meta`
- `--inner_updates`
- `--query_updates` (mainly for MAML)
- `--meta_lr`
- `--task_families` (comma-separated subset)

Suggested starting points:
- MAML: `tasks_per_meta=4`, `inner_updates=1`, `query_updates=1`, `meta_lr=0.03-0.06`
- Reptile: `tasks_per_meta=4-8`, `inner_updates=2`, `meta_lr=0.05-0.1`

## Notes

- The MAML implementation here is first-order and intentionally lightweight for practical use with skrl PPO.
- If you later want exact higher-order gradients, that requires a fully differentiable inner optimizer path (substantially heavier).
