#!/usr/bin/env bash
set -euo pipefail

# cd "$(dirname "$0")/../.."
for seed in {3..10}; 
do
	echo "[train_symm_aug_ppo] Running seed=${seed}"
	python train.py --task Quadcontrol-SymmAug-v0 --seed "${seed}" \
		--headless --video --max_iterations 300
done
