CONFIG=_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048_aff192x256_hrFix_lvl3_ratio05_vidmaxpix/3B_vqvaeStep340000/train.yaml
LOG=_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048_aff192x256_hrFix_lvl3_ratio05_vidmaxpix/3B_vqvaeStep340000/train.log


# CUDA_VISIBLE_DEVICES=0,1,3,4 \
#     nohup \
#     llamafactory-cli \
#     train \
#     $CONFIG \
#     > $LOG &

# CUDA_VISIBLE_DEVICES=3,4 \
#     llamafactory-cli \
#     train \
#     $CONFIG

ALLOW_EXTRA_ARGS=true \
CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node 1 --master_port 28054 -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG