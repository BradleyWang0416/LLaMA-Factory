CONFIG=examples/new_byBradley/train_lora/_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048/merge.yaml

CUDA_VISIBLE_DEVICES=0 \
    llamafactory-cli \
    export \
    $CONFIG