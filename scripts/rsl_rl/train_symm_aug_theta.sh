#!/usr/bin/env bash
set -euo pipefail

discretisations=(4 6 8 12)
seeds=(0 1 2)

for discretisation in "${discretisations[@]}"; do
  for seed in "${seeds[@]}"; do
    run_name="c_${discretisation}/seed_${seed}"
    echo "[train_symm_aug_theta] discretisation=${discretisation} seed=${seed}"
    python train.py --task Quadcontrol-SymmAugTheta-v0 --seed "${seed}" \
      --run_name "${run_name}" \
      --headless --video --max_iterations 300 \
      env.yaw_discretisations="${discretisation}"
  done
done