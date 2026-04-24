#!/usr/bin/env bash
set -euo pipefail

# Sweep one parameter only: algorithm.symmetry_cfg.mirror_loss_coeff
coeffs=(0.05 0.20 1.00 2.00)

for coeff in "${coeffs[@]}"; do
    for seed in 0 1 2; do
        run_name="mlc_${coeff//./p}_seed_${seed}"
        echo "[train_mirror_loss_coeff_sweep] coeff=${coeff} seed=${seed}"
        python train.py --task Quadcontrol-MirrorLoss-v0 "$@" --seed "${seed}" \
            --run_name "${run_name}" \
            --headless --video --max_iterations 300 \
            agent.algorithm.symmetry_cfg.mirror_loss_coeff="${coeff}"
    done
done
