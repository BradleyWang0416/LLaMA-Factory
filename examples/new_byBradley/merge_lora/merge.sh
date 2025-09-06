CONFIG=examples/new_byBradley/merge_lora/qwen2_5vl_lora_sft_byBrad_vid2skel.yaml

CUDA_VISIBLE_DEVICES=5 \
    llamafactory-cli \
    export \
    $CONFIG