CONFIG=examples/new_byBradley/train_lora/_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048/infer.yaml

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5

CUDA_VISIBLE_DEVICES=7 \
    torchrun \
    -m debugpy --listen 5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG