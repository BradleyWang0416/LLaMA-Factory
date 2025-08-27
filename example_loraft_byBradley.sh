# CONFIG=examples/train_lora/qwen2_5vl_lora_sft.yaml
CONFIG=examples/train_lora/qwen2_5vl_lora_sft_byBrad.yaml


CUDA_VISIBLE_DEVICES=5 \
    torchrun --nproc_per_node 1 -m debugpy --listen 5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG