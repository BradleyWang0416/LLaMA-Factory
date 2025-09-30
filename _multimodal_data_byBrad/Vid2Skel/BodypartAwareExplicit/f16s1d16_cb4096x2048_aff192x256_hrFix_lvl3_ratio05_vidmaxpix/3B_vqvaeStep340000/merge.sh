CONFIG=_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/f16s1d16_cb4096x2048_aff192x256_hrFix_lvl3_ratio05_vidmaxpix/3B_vqvaeStep340000/merge.yaml

CUDA_VISIBLE_DEVICES=3 \
    llamafactory-cli \
    export \
    $CONFIG