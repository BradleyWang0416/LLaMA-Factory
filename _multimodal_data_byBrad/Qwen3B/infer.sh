CONFIG=_multimodal_data_byBrad/Qwen3B/infer.yaml


export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=5

CUDA_VISIBLE_DEVICES=3 \
    torchrun --master_port 28054 \
    -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    src/llamafactory/launcher.py \
    $CONFIG