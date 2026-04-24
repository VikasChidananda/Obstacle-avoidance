#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
for seed in 0 1 2; do
    echo "[train_naive_ppo] Running seed=${seed}"
    python train.py --task Quadcontrol-v0 \
        --headless --video --seed $seed
done
