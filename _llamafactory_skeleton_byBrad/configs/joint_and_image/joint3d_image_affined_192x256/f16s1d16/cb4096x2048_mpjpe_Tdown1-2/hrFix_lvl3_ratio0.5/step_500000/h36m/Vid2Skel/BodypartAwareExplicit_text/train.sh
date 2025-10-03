#!/bin/bash
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "$SCRIPT_DIR"

CONFIG_NAME=train_3B.yaml
LOG_NAME=train.log

CONFIG="${SCRIPT_DIR}/${CONFIG_NAME}"
LOG="${SCRIPT_DIR}/${LOG_NAME}"


# CUDA_VISIBLE_DEVICES=0,1 \
    # nohup \
    # llamafactory-cli \
    # train \
    # $CONFIG \
    # > $LOG &

# CUDA_VISIBLE_DEVICES=0,1 \
    # llamafactory-cli \
    # train \
    # $CONFIG

ALLOW_EXTRA_ARGS=true \
CUDA_VISIBLE_DEVICES=7 \
    torchrun --nproc_per_node 1 -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG