CONFIG=_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048_aff192x256/hrFix_lvl3_ratio05/train.yaml
LOG=_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048_aff192x256/hrFix_lvl3_ratio05/train.log


export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5


# CUDA_VISIBLE_DEVICES=3,4,5,6 \
#     nohup \
#     llamafactory-cli \
#     train \
#     $CONFIG \
#     > $LOG &

ALLOW_EXTRA_ARGS=true \
CUDA_VISIBLE_DEVICES=5 \
    torchrun --nproc_per_node 1 -m debugpy --listen 5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG

# ALLOW_EXTRA_ARGS=true \
# CUDA_VISIBLE_DEVICES=6 \
#     llamafactory-cli train \
#     $CONFIG