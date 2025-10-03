# mode=debug
mode=infer


save_root=_llamafactory_skeleton_byBrad/data/joint_and_image
save_subdir_raw=joint3d_image_affined_192x256/f16s1d16
save_subdir_vqvae=cb8192x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5/step_300000
# save_subdir_jsonl=Vid2Skel/BodypartAwareExplicit
save_subdir_jsonl=h36m/Vid2Skel


############################# dataset config #######################################################
data_split=train
num_frame=16
sample_stride=1
data_stride=16
return_extra="[['image']]"
get_item_list="['joint3d_image','joint3d_image_normed','factor_2_5d','joint3d_image_scale','joint3d_image_transl','video_rgb','joint3d_image_affined','joint3d_image_affined_normed','joint3d_image_affined_scale','joint3d_image_affined_transl','slice_id','image_sources','joint_2_5d_image','affine_trans','affine_trans_inv','joint2d','joint2d_cpn','joint3d_cam','joint3d_cam_rootrel_meter']"
save_item_list="[\
'joint3d_image_affined_normed','joint3d_image_affined_scale','joint3d_image_affined_transl',\
'joint3d_image_affined',\
'joint3d_image_scale','joint3d_image_transl',\
'joint_2_5d_image','joint2d','joint2d_cpn','joint3d_cam',\
'factor_2_5d','affine_trans','affine_trans_inv',\
]"

load_data_file="wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final_wImgPath_wJ3dCam_wJ2dCpn.pkl"
load_image_source_file="wxs/DATASETS/Human3.6M_for_MotionBERT/images_source.pkl"
load_bbox_file="wxs/DATASETS/Human3.6M_for_MotionBERT/bboxes_xyxy.pkl"
load_text_source_file=''

normalize=anisotropic
filter_invalid_images=True
processed_image_shape="[192,256]"
backbone=hrnet_32


############################# vqvae config #######################################################
joint_data_type=joint3d_image_affined_normed
resume_path=../MTVCrafter_weights/vqvae_experiment/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb8192x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5/models/checkpoint_epoch_395_step_300000
# resume_path=../MTVCrafter/vqvae_experiment/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb8192x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5/models/checkpoint_epoch_395_step_300000
num_code=8192   # 8192
code_dim=2048   # 3072
batch_size=16
hrnet_output_level=3    # int or list. 0,1,2,3 分别对应输出 [B,32,H/4,W/4], [B,64,H/8,W/8], [B,128,H/16,W/16], [B,256,H/32,W/32] 的特征
vision_guidance_ratio=0.5
downsample_time="[1,2]"
frame_upsample_rate="[2.0,1.0]"


############################# jsonl config #######################################################
prompt_template=BodypartAwareExplicit_text
task=Vid2Skel




if [ "$mode" = "debug" ]; then
    DEBUG_ARGS="-m debugpy --listen 0.0.0.0:5678 --wait-for-client"
else
    DEBUG_ARGS="-u"
fi



CUDA_VISIBLE_DEVICES=0 \
    python \
    $DEBUG_ARGS \
    _llamafactory_skeleton_byBrad/data_utils/generate_multimodal_data.py \
    --save_root ${save_root} \
    --save_subdir_raw ${save_subdir_raw} \
    --save_subdir_vqvae ${save_subdir_vqvae} \
    --save_subdir_jsonl ${save_subdir_jsonl} \
    \
    --data_split ${data_split} \
    --num_frames ${num_frame} \
    --sample_stride ${sample_stride} \
    --data_stride ${data_stride} \
    --return_extra ${return_extra} \
    --get_item_list ${get_item_list} \
    --save_item_list ${save_item_list} \
    --load_data_file "${load_data_file}" \
    --load_image_source_file "${load_image_source_file}" \
    --load_bbox_file "${load_bbox_file}" \
    --load_text_source_file "${load_text_source_file}" \
    --normalize ${normalize} \
    --filter_invalid_images ${filter_invalid_images} \
    --processed_image_shape ${processed_image_shape} \
    \
    --joint_data_type ${joint_data_type} \
    --resume_pth "${resume_path}" \
    --nb_code ${num_code} \
    --codebook_dim ${code_dim} \
    --batch_size ${batch_size} \
    --backbone ${backbone} \
    --hrnet_output_level ${hrnet_output_level} \
    --vision_guidance_ratio ${vision_guidance_ratio} \
    --downsample_time ${downsample_time} \
    --frame_upsample_rate ${frame_upsample_rate} \
    \
    --prompt_template ${prompt_template} \
    --task ${task}