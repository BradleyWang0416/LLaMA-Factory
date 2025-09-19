CONFIG=examples/new_byBradley/infer/qwen2_5vl_lora_sft_byBrad_all3.yaml

CUDA_VISIBLE_DEVICES=0 \
    torchrun \
    -m debugpy --listen 5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG