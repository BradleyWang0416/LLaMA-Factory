CONFIG=examples/new_byBradley/infer/qwen2_5vl_lora_sft_byBrad_vid2skel.yaml

CUDA_VISIBLE_DEVICES=6 \
    torchrun \
    -m debugpy --listen 5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG