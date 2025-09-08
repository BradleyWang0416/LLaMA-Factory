CONFIG=examples/new_byBradley/infer/qwen2_5vl_lora_sft_byBrad_M0102.yaml

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
    torchrun \
    -m debugpy --listen 5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG