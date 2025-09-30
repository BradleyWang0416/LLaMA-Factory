CONFIG=_multimodal_data_byBrad/Vid2Skel/BodypartAwareExplicit/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb4096x2048_mpjpe_Tdown1-2/hrFixExclStage4.2_lvl3_ratio0.5/step300000/merge.yaml

CUDA_VISIBLE_DEVICES=0 \
    llamafactory-cli \
    export \
    $CONFIG