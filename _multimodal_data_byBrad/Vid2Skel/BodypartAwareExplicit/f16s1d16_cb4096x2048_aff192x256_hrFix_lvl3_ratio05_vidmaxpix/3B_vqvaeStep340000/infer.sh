CONFIG=_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048_aff192x256_hrFix_lvl3_ratio05_vidmaxpix/3B_vqvaeStep340000/infer.yaml

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5

CUDA_VISIBLE_DEVICES=3 \
    torchrun --master_port 28054 \
    -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG