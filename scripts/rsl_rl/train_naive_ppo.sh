#!/usr/bin/env bash
set -euo pipefail
experiment_name="naive_ppo"

for seed in {3..10}; 
do
    echo "[train_naive_ppo] Running seed=${seed}"
    python train.py --task Quadcontrol-v0 \
        --headless --video --seed ${seed} \
        --experiment ${experiment_name} \
        --max_iterations 300
done