# CONFIG=examples/new_byBradley/infer/joint_aware_explicit/vid2skel_f16s1d16.yaml
CONFIG=examples/new_byBradley/infer/joint_aware_explicit/wMPJPE/vid2skel_f16s1d16.yaml

# CONFIG=examples/new_byBradley/infer/bodypart_aware_explicit/vid2skel_f16s1d16.yaml

export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5

CUDA_VISIBLE_DEVICES=0 \
    torchrun \
    -m debugpy --listen 5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG