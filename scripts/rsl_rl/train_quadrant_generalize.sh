#!/usr/bin/env bash
# Train an agent with C4 augmentation + mirror loss on 1st-quadrant goals,
# then evaluate generalisation to the full 3D goal space (all four quadrants).
#
# Usage:
#   bash scripts/rsl_rl/train_quadrant_generalize.sh [rsl_rl train.py args...]
#
# Examples:
#   bash scripts/rsl_rl/train_quadrant_generalize.sh --headless --num_envs 4096
#   bash scripts/rsl_rl/train_quadrant_generalize.sh --headless --max_iterations 300

set -euo pipefail

cd "$(dirname "$0")"

TRAIN_TASK="Quadcontrol-QuadrantGoal-SymmAugMirror-v0"
EVAL_TASK="Quadcontrol-Full3DGoal-SymmAugMirror-v0"
EXPERIMENT_NAME="symmetry/quadrant_generalize"

echo "=============================================================="
echo " Step 1/2: Training on 1st-quadrant goals (${TRAIN_TASK})"
echo "=============================================================="

python train.py --task "${TRAIN_TASK}" "$@"

echo ""
echo "=============================================================="
echo " Step 2/2: Evaluating generalisation to full 3D goals"
echo "=============================================================="

python eval_quadrant_generalize.py --task "${EVAL_TASK}" "$@"
