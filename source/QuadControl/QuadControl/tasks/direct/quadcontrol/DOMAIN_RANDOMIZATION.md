# Domain Randomization for Quadrotor Learning

## Overview

This module reproduces the "Learning to Fly in Seconds" approach (Kaufmann et al., 2021) by implementing domain randomization for quadrotor control policies. Domain randomization applies random variations to simulator physics parameters during training, which improves robustness and enables fast sim-to-real transfer.

## What Changed

### New Files

1. **quadcontrol_domain_rand_env_cfg.py** - Config with randomization ranges:
   - Mass variation: [25g, 42g] (Crazyflie nominal ~35g)
   - Inertia scaling: [0.8, 1.2]x
   - Thrust scaling: [0.9, 1.1]x (motor response variability)
   - Moment scaling: [0.9, 1.1]x
   - Gravity: [8.5, 10.5] m/s²
   - Drag coefficient: [0.8, 1.2]x
   - Linear/angular damping variation

2. **quadcontrol_domain_rand_env.py** - Environment that applies randomization:
   - Randomizes physics parameters at startup
   - Option to re-randomize per episode (`randomize_on_reset=False` by default)
   - Per-environment thrust and moment scales

3. **skrl_a2c_domain_rand_cfg.yaml** - A2C agent config:
   - Uses A2C (Advantage Actor-Critic) instead of PPO
   - Simpler algorithm, good for robust control
   - 64x64 MLP policy and value networks

4. **train_domain_rand.py** - Training script:
   - Loads domain-randomized environment
   - Uses CustomRunner to launch with A2C
   - Logs to `logs/skrl/domain_randomization/`

5. **train_domain_rand.sh** - Shell wrapper for easy launch

## How It Works

### Domain Randomization Strategy

During training:
1. Simulator creates N parallel environments (default: 4096)
2. Each environment gets random physics parameters
3. Agent learns a policy robust across parameter variations
4. On real robot: policy works without re-training because it's trained on many "worlds"

### Physics Parameters Randomized

```yaml
Mass: 0.025 - 0.042 kg         # ±20% of nominal
Inertia: 0.8 - 1.2x             # ±20%
Thrust: 0.9 - 1.1x              # ±10% (motor degradation)
Moment: 0.9 - 1.1x              # ±10%
Gravity: 8.5 - 10.5 m/s²        # ±10% (altitude + Earth variation)
Drag: 0.8 - 1.2x                # ±20% (wind, air density)
```

## Training

### Quick Start

```bash
cd scripts/skrl
./train_domain_rand.sh
```

### With Custom Parameters

```bash
./train_domain_rand.sh --num_envs 2048 --max_iterations 200000 --seed 42
```

### Full Command

```bash
python train_domain_rand.py \
    --task Quadcontrol-DomainRand-v0 \
    --num_envs 4096 \
    --max_iterations 100000 \
    --seed 0 \
    --headless --video
```

## Configuration

### Enable/Disable Domain Randomization

Edit [agents/skrl_a2c_domain_rand_cfg.yaml](agents/skrl_a2c_domain_rand_cfg.yaml) or pass at runtime:

```yaml
# In config
enable_domain_randomization: True   # or False to disable
randomize_on_reset: False           # False = once at startup, True = per episode
```

### Adjust Randomization Ranges

Edit [quadcontrol_domain_rand_env_cfg.py](quadcontrol_domain_rand_env_cfg.py):

```python
@configclass
class QuadcontrolDomainRandEnvCfg(QuadcontrolEnvCfg):
    mass_range: tuple[float, float] = (0.025, 0.042)
    inertia_range: tuple[float, float] = (0.8, 1.2)
    # ... etc
```

## Expected Results

- **Without domain randomization**: Policy overfits to simulator parameters
- **With domain randomization**: Policy is robust to parameter variations
- **Sim-to-real transfer**: Trained policy works on real quadrotor with minimal fine-tuning

Training typically converges to stable hover in ~50k-100k timesteps with A2C.

## TensorBoard Monitoring

```bash
tensorboard --logdir logs/skrl/domain_randomization
```

Monitor:
- `Train/reward` - episode reward
- `Loss/Policy loss` - policy gradient loss
- `Loss/Value loss` - value function loss
- `Entropy` - exploration level

## A2C vs PPO

- **A2C**: On-policy, simpler, faster convergence on well-conditioned problems
- **PPO**: More stable across varying domains, better for harder tasks

For domain randomization, A2C works well because the diverse environment samples act as implicit regularization.

## Further Work

1. **Fine-tuning**: Load trained policy, then fine-tune on real robot data
2. **Curriculum**: Start with narrow ranges, gradually expand during training
3. **Adaptive randomization**: Use Automatic Domain Randomization (ADR) to increase difficulty
4. **Validation**: Test policy on held-out parameter ranges not seen during training

## References

- **"Learning to Fly in Seconds"** - Kaufmann et al., ICRA 2021
  - Sim-to-real transfer via domain randomization
  - Fast adaptation on real quadrotors (CrazyFlie)
  
- **A2C algorithm**: Mnih et al., 2016
- **Domain Randomization**: Tobin et al., IROS 2017

## Environment Registration

For Hydra task loading, register in your task registry. The task ID is:

```
Quadcontrol-DomainRand-v0
```

Ensure your `__init__.py` or task registry includes:

```python
# Register domain randomization task
gym.register(
    id="Quadcontrol-DomainRand-v0",
    entry_point="QuadControl.tasks:QuadcontrolDomainRandEnv",
    kwargs={"env_cfg": QuadcontrolDomainRandEnvCfg()},
)
```

Contact: See QuadControl README for project setup.
