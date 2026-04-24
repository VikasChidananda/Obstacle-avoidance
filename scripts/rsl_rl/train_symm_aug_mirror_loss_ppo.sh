#!/usr/bin/env bash
set -euo pipefail

# cd "$(dirname "$0")/../.."
for seed in {0..10}; do
	echo "[train_symm_aug_mirror_loss_ppo] Running seed=${seed}"
	python train.py --task Quadcontrol-SymmAugMirror-v0 --headless --video "$@" --seed "${seed}" \
		--max_iterations 300
done
