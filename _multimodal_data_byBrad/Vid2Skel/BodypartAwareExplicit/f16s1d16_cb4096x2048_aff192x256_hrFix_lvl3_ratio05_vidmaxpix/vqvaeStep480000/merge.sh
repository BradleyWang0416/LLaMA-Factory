CONFIG=_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048_aff192x256_hrFix_lvl3_ratio05_vidmaxpix/vqvaeStep480000/merge.yaml

CUDA_VISIBLE_DEVICES=0 \
    llamafactory-cli \
    export \
    $CONFIG