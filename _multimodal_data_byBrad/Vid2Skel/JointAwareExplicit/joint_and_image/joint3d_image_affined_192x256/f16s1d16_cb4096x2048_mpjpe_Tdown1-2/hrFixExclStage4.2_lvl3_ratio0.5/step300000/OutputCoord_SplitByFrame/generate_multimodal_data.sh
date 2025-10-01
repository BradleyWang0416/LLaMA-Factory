# mode=debug
mode=infer

downsample_time="[1,2]"
frame_upsample_rate="[2.0,1.0]"

joint_data_type=joint3d_image_affined_normed
get_item_list="['joint3d_image','joint3d_image_normed','factor_2_5d','joint3d_image_scale','joint3d_image_transl','video_rgb','joint3d_image_affined','joint3d_image_affined_normed','joint3d_image_affined_scale','joint3d_image_affined_transl','slice_id','image_sources','joint_2_5d_image','affine_trans','affine_trans_inv','joint2d','joint2d_cpn','joint3d_cam','joint3d_cam_rootrel_meter']"


prompt_template=JointAwareExplicit
task=Vid2Skel
data_split=test
# resume_path=../MTVCrafter_weights/vqvae_experiment/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb4096x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5/models/checkpoint_epoch_185_step_300000
resume_path=../MTVCrafter/vqvae_experiment/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb4096x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5/models/checkpoint_epoch_185_step_300000

BATCH_SIZE=32

DATA_MODE=joint3d

NUM_CODE=4096   # 8192
CODE_DIM=2048   # 3072
NUM_FRAME=16
SAMPLE_STRIDE=1
DATA_STRIDE=16
processed_image_shape="[192,256]"
HRNET_OUTPUT_LEVEL=3    # int or list. 0,1,2,3 分别对应输出 [B,32,H/4,W/4], [B,64,H/8,W/8], [B,128,H/16,W/16], [B,256,H/32,W/32] 的特征
VISION_GUIDANCE_RATIO=0.5

save_dir=joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb4096x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5/step_300000
# save_dir=f${NUM_FRAME}s${SAMPLE_STRIDE}d${DATA_STRIDE}_cb${NUM_CODE}x${CODE_DIM}



# load_data_file="../Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final_wImgPath_wJ3dCam_wJ2dCpn.pkl"
# load_image_source_file=../Human3.6M_for_MotionBERT/images_source.pkl
# load_bbox_file=../Human3.6M_for_MotionBERT/bboxes_xyxy.pkl
load_data_file="/data1/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final_wImgPath_wJ3dCam_wJ2dCpn.pkl"
load_image_source_file="/data1/wxs/DATASETS/Human3.6M_for_MotionBERT/images_source.pkl"
load_bbox_file="/data1/wxs/DATASETS/Human3.6M_for_MotionBERT/bboxes_xyxy.pkl"
load_text_source_file=''
return_extra="[['image']]"
# data preprocessing config
normalize=anisotropic
# image config
filter_invalid_images=True
backbone=hrnet_32



if [ "$mode" = "debug" ]; then
    DEBUG_ARGS="-m debugpy --listen 0.0.0.0:5678 --wait-for-client"
else
    DEBUG_ARGS="-u"
fi



CUDA_VISIBLE_DEVICES=3 \
    python \
    $DEBUG_ARGS \
    src/llamafactory/extras_byBrad/generate_multimodal_data.py \
    --joint_data_type ${joint_data_type} \
    --get_item_list ${get_item_list} \
    --prompt_template ${prompt_template} \
    --task ${task} \
    --data_split ${data_split} \
    --save_dir ${save_dir} \
    --resume_pth "${resume_path}" \
    --batch_size ${BATCH_SIZE} \
    --data_mode ${DATA_MODE} \
    --nb_code ${NUM_CODE} \
    --codebook_dim ${CODE_DIM} \
    --num_frames ${NUM_FRAME} \
    --sample_stride ${SAMPLE_STRIDE} \
    --data_stride ${DATA_STRIDE} \
    --hrnet_output_level ${HRNET_OUTPUT_LEVEL} \
    --vision_guidance_ratio ${VISION_GUIDANCE_RATIO} \
    --load_data_file "${load_data_file}" \
    --load_image_source_file "${load_image_source_file}" \
    --load_bbox_file "${load_bbox_file}" \
    --load_text_source_file "${load_text_source_file}" \
    --return_extra ${return_extra} \
    --normalize ${normalize} \
    --filter_invalid_images ${filter_invalid_images} \
    --processed_image_shape ${processed_image_shape} \
    --backbone ${backbone} \
    --downsample_time ${downsample_time} \
    --frame_upsample_rate ${frame_upsample_rate}