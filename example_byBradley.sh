    # python scripts/vllm_infer.py \
CUDA_VISIBLE_DEVICES=5 \
    python -m debugpy --listen 5678 --wait-for-client scripts/vllm_infer.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --template qwen2_vl \
    --dataset mllm_demo