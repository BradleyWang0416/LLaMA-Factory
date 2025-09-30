CONFIG=_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb4096x2048_mpjpe_Tdown1-2/hrFixExclStage4.2_lvl3_ratio0.5/step300000/vidmaxpix49168_infer.yaml

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5

CUDA_VISIBLE_DEVICES=3 \
    torchrun --master_port 29025 \
    -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG