CONFIG=examples/new_byBradley/merge_lora/qwen2_5vl_lora_sft_byBrad_all3.yaml

CUDA_VISIBLE_DEVICES=4 \
    llamafactory-cli \
    export \
    $CONFIG