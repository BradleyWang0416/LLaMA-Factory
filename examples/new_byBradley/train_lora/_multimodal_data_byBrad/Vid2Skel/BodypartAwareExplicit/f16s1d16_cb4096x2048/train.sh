CONFIG=examples/new_byBradley/train_lora/_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048/train.yaml
LOG=examples/new_byBradley/train_lora/_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048/train.log


# CUDA_VISIBLE_DEVICES=3,4,5,6 \
#     nohup \
#     llamafactory-cli \
#     train \
#     $CONFIG \
#     > $LOG &

ALLOW_EXTRA_ARGS=true \
CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node 1 -m debugpy --listen 5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG