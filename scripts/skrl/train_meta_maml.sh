#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/train_meta.py" --meta_algo maml --headless --video --max_iterations 300 \
    --experiment "meta_maml" \
    --tasks_per_meta 10 \
    --inner_updates 2 \
    --meta_lr 0.001 \
    --seed 0 \
    "$@"
