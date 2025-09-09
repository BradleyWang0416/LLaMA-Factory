# CONFIG=examples/new_byBradley/train_lora/qwen2_5vl_lora_sft_byBrad_skelPred_f16s2d16.yaml
# LOG=exp_skelPred_f16s2d16.log

# CONFIG=examples/new_byBradley/train_lora/qwen2_5vl_lora_sft_byBrad_text2skel_f64s2d64.yaml
# LOG=exp_text2skel_f64s2d64.log

# CONFIG=examples/new_byBradley/train_lora/qwen2_5vl_lora_sft_byBrad_vid2skel_f16s2d16.yaml
# LOG=exp_vid2skel_f16s2d16.log

# CONFIG=examples/new_byBradley/train_lora/NoBodyPartTag/skelPred_f16s2d16.yaml
# LOG=exp_skelPred_f16s2d16_NoBodyPartTag.log

# CONFIG=examples/new_byBradley/train_lora/NoBodyPartTag/text2skel_f64s2d64.yaml
# LOG=exp_text2skel_f64s2d64_NoBodyPartTag.log

CONFIG=examples/new_byBradley/train_lora/NoBodyPartTag/vid2skel_f16s2d16.yaml
LOG=exp_vid2skel_f16s2d16_NoBodyPartTag.log


CUDA_VISIBLE_DEVICES=6 \
    nohup \
    llamafactory-cli \
    train \
    $CONFIG \
    > $LOG &

# ALLOW_EXTRA_ARGS=true \
# CUDA_VISIBLE_DEVICES=6 \
#     torchrun --nproc_per_node 1 -m debugpy --listen 5678 --wait-for-client \
#     src/llamafactory/launcher.py \
#     $CONFIG