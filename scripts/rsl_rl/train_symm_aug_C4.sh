#!/usr/bin/env bash
set -euo pipefail


for seed in {0..10}; do
  echo "[train_symm_aug_c4_ppo] Running seed=${seed}"
  python train.py --task Quadcontrol-SymmAugC4-v0 --seed "${seed}" \
    --headless --video --max_iterations 300
done
