CONFIG=examples/new_byBradley/train_lora/qwen2_5vl_lora_sft_byBrad_vid2skel.yaml

# CUDA_VISIBLE_DEVICES=5,6 \
#     llamafactory-cli \
#     train \
#     $CONFIG

ALLOW_EXTRA_ARGS=true \
CUDA_VISIBLE_DEVICES=6 \
    torchrun --nproc_per_node 1 -m debugpy --listen 5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG