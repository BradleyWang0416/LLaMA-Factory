#!/bin/bash
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "$SCRIPT_DIR"

CONFIG="${SCRIPT_DIR}/infer.yaml"

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5

CUDA_VISIBLE_DEVICES=3 \
    torchrun --master_port 29025 \
    -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG