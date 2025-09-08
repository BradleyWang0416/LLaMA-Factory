CONFIG=examples/new_byBradley/merge_lora/qwen2_5vl_lora_sft_byBrad_M0102.yaml

CUDA_VISIBLE_DEVICES=7 \
    llamafactory-cli \
    export \
    $CONFIG