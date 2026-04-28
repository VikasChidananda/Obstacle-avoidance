#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/train_domain_rand.py" \
    --headless \
    --video \
    --max_iterations 100000 \
    --experiment "domain_randomization" \
    "$@"
