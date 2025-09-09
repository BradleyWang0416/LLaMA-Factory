CONFIG=examples/new_byBradley/infer/NoBodyPartTag/skelPred_f16s2d16.yaml

CUDA_VISIBLE_DEVICES=3 \
    torchrun \
    -m debugpy --listen 5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG