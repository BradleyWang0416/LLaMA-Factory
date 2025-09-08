CONFIG=examples/new_byBradley/train_lora/qwen2_5vl_lora_sft_byBrad_all3.yaml

CUDA_VISIBLE_DEVICES=1,2,5,6 \
    nohup \
    llamafactory-cli \
    train \
    $CONFIG \
    > exp_all3.log &

# ALLOW_EXTRA_ARGS=true \
# CUDA_VISIBLE_DEVICES=6 \
#     torchrun --nproc_per_node 1 -m debugpy --listen 5678 --wait-for-client \
#     src/llamafactory/launcher.py \
#     $CONFIG