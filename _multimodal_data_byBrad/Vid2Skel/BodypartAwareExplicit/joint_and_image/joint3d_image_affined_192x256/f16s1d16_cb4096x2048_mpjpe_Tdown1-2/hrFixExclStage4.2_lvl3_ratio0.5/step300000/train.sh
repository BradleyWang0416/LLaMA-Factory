EXP_DIR=_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb4096x2048_mpjpe_Tdown1-2/hrFixExclStage4.2_lvl3_ratio0.5/step300000

CONFIG_NAME=vidmaxpix49168_train.yaml
LOG_NAME=vidmaxpix49168_train.log

CONFIG="${EXP_DIR}/${CONFIG_NAME}"
LOG="${EXP_DIR}/${LOG_NAME}"


# CUDA_VISIBLE_DEVICES=0,1 \
    nohup \
    llamafactory-cli \
    train \
    $CONFIG \
    > $LOG &

# CUDA_VISIBLE_DEVICES=0,1 \
    # llamafactory-cli \
    # train \
    # $CONFIG

# ALLOW_EXTRA_ARGS=true \
# CUDA_VISIBLE_DEVICES=0 \
#     torchrun --nproc_per_node 1 -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
#     src/llamafactory/launcher.py \
#     $CONFIG