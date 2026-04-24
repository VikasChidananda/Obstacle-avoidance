#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
for seed in 0 1 2; do
	echo "[train_symm_aug_mirror_loss_ppo] Running seed=${seed}"
	python scripts/rsl_rl/train.py --task Quadcontrol-SymmAugMirror-v0 --headless --video "$@" --seed "${seed}"
done
