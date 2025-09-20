import os
import os.path as osp
import torch
import numpy as np
import joblib
import easydict as edict
from tqdm import tqdm
from PIL import Image
import json 
import random
random.seed(0)
import codecs as cs
import ast
import argparse
from safetensors.torch import load_file as load_safetensors
from collections import defaultdict

from llamafactory.extras_byBrad.vqvae import SKEL_VQVAE, Encoder, VectorQuantizer, Decoder

import sys
sys.path.append("/home/wxs/ContextAwarePoseFormer_Private/H36M-Toolbox/")
from multimodal_h36m_dataset_byBradley import Multimodal_Mocap_Dataset
sys.path.remove("/home/wxs/ContextAwarePoseFormer_Private/H36M-Toolbox/")
sys.path.append('/home/wxs/Skeleton-in-Context-tpami/')
from lib.utils.viz_skel_seq import viz_skel_seq_anim
sys.path.remove('/home/wxs/Skeleton-in-Context-tpami/')
sys.path.append('/home/wxs/MTVCrafter/')
from config.vision_backbone import config as vision_config
from config.vqvae import vqvae_config
from models import HYBRID_VQVAE
sys.path.remove('/home/wxs/MTVCrafter/')


PROMPT_TEMPLATES = {
    'Vid2Skel': {
        'fixed': [
            "Generate the skeleton sequence for the video <video>."
        ],
        'simple': [
            # --- 简洁格式说明 (大部分采用此形式) ---
            "Generate the skeleton sequence for the video <video>.",
            "Transcribe the actions in <video> into a sequence of skeleton tokens.",
            "What is the skeleton token sequence for the person in <video>?",
            "You are a motion analysis expert. Analyze the video <video> and output the corresponding motion data.",
            "Here is a video of a person moving: <video>. The corresponding skeletal representation is:",
            "Analyze the person's movement in <video> and represent it using skeleton tokens.",

            # --- 完整格式说明 (作为清晰的“锚点”) ---
            "Please provide the skeletal representation for the movement in the video <video>.",
        ],
        'bodypart_aware': [
            # --- 简洁格式说明 (大部分采用此形式) ---
            "Generate the skeleton sequence for the video <video>. Please structure the output for each frame using body part tags (e.g., <torso>...</torso>, <left_arm>...</left_arm>, etc.).",
            "Transcribe the actions in <video> into a sequence of skeleton tokens, ensuring the output is formatted by body parts for each frame.",
            "What is the skeleton token sequence for the person in <video>? Please provide the answer in a structured format with body part tags.",
            "You are a motion analysis expert. Analyze the video <video> and output the corresponding motion data. Your output must be structured frame-by-frame, with each frame containing body part sections like <torso>...</torso>.",
            "Here is a video of a person moving: <video>. The corresponding structural representation, organized by body parts, is:",
            "Analyze the person's movement in <video> and represent it structurally using body part tags.",

            # --- 完整格式说明 (作为清晰的“锚点”) ---
            "Please provide the skeletal representation for the movement in the video <video>. Your response must be structured for each frame using all five body part tags: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
            "What would the motion capture data for the video <video> look like? Output the data using the precise format for each frame: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
        ],
        'BodypartAwareExplicit': [
            "Please provide the skeletal representation for the movement in the video <video>. Use <|frame_break|> to separate frames. Your response must be structured for each frame using all five body part tags: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
        ],
        'joint_aware_explicit': [
            "Please provide the skeletal representation for the movement in the video <video>. Use <|frame_break|> to separate frames. "
            "Your response must be structured for each frame using all 17 joint tags: <Hips>...</Hips><Right_Hip>...</Right_Hip><Right_Knee>...</Right_Knee><Right_Foot>...</Right_Foot><Left_Hip>...</Left_Hip><Left_Knee>...</Left_Knee><Left_Foot>...</Left_Foot><Spine>...</Spine><Thorax>...</Thorax><Neck/Nose>...</Neck/Nose><Head>...</Head><Left_Shoulder>...</Left_Shoulder><Left_Elbow>...</Left_Elbow><Left_Wrist>...</Left_Wrist><Right_Shoulder>...</Right_Shoulder><Right_Elbow>...</Right_Elbow><Right_Wrist>...</Right_Wrist>.",
        ],
    },
    'SkelPred': {
        'fixed': [
            "Predict the future motion based on the provided skeleton sequence <skeleton>.",
        ],
        'simple': [
            # --- 简洁格式说明 (大部分采用此形式) ---
            "Continue the motion sequence provided in <skeleton>.",
            "Predict the future motion based on the provided skeleton sequence <skeleton>.",
            "Given the motion <skeleton>, what happens next?",
            "You are a motion prediction expert. Analyze the past motion <skeleton> and output the most likely future motion.",
            "Here is the beginning of a motion sequence: <skeleton>. The continuation of the motion is:",
            "Generate the next set of skeleton tokens that logically follow this sequence: <skeleton>.",

            # --- 完整格式说明 (作为清晰的"锚点") ---
            "Past motion: <skeleton>. Future motion:",
        ],
        'bodypart_aware': [
            # --- 简洁格式说明 ---
            "Continue the motion sequence provided in <skeleton>. The predicted motion should follow the same body part structure (e.g., <torso>...</torso>).",
            "Predict the future motion based on the provided skeleton sequence <skeleton>. Generate the next set of skeleton tokens using the established body part format.",
            "Given the motion <skeleton>, what happens next? Please provide the answer in the same structural format.",
            "You are a motion prediction expert. Analyze the past motion <skeleton> and output the most likely future motion, maintaining the structural format.",
            "Here is the beginning of a motion sequence: <skeleton>. The continuation of the motion, maintaining the same structure, is:",
            
            # --- 完整格式说明 (作为清晰的“锚点”) ---
            "Generate the next set of skeleton tokens that logically follow this sequence: <skeleton>. Ensure the output adheres to the full structure: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
            "Past motion: <skeleton>. Future motion (formatted with all body part tags):",
        ],
        'BodypartAwareExplicit': [
            "Generate the next set of skeleton tokens that logically follow this sequence: <skeleton>. Use <|frame_break|> to separate frames. Ensure the output adheres to the full structure: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
        ],
        'joint_aware_explicit': [
            # TODO
        ],
    },
    'Text2Skel': {
        'fixed': [
            "Generate the skeleton motion for the description: \"<text_description>\".",
        ],
        'simple': [
            # --- 简洁格式说明 (大部分采用此形式) ---
            "Generate the skeleton sequence for the following description: <text_description>.",
            "Create a motion sequence based on this text: \"<text_description>\".",
            "What would the motion for '<text_description>' look like in skeleton tokens?",
            "You are a choreographer. Animate the following description into a skeleton sequence: <text_description>.",
            "Description: <text_description>. Corresponding skeleton sequence:",
            "Convert the following text into skeleton motion: \"<text_description>\".",

            # --- 完整格式说明 (作为清晰的"锚点") ---
            "Generate the skeleton motion for the description: \"<text_description>\".",
        ],
        'bodypart_aware': [
            # --- 简洁格式说明 ---
            "Generate the skeleton sequence for the following description: <text_description>. Please structure the output using body part tags (e.g., <torso>...</torso>).",
            "Create a motion sequence based on this text: \"<text_description>\". Ensure the output is formatted with body part tags for each frame.",
            "What would the motion for '<text_description>' look like in skeleton tokens? Please provide the answer in the structured body part format.",
            "You are a choreographer. Animate the following description into a skeleton sequence: <text_description>. Use the standard body part structure for your output.",
            "Description: <text_description>. Corresponding skeleton sequence:",

            # --- 完整格式说明 (作为清晰的“锚点”) ---
            "Generate the skeleton motion for the description: \"<text_description>\". Your response must be structured for each frame using all five body part tags: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
        ],
        'BodypartAwareExplicit': [
            "Generate the skeleton motion for the description: \"<text_description>\". Use <|frame_break|> to separate frames. Your response must be structured for each frame using all five body part tags: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
        ],
        'joint_aware_explicit': [
            # TODO
        ],
    },
}

TASK_TEMPLATE = {
    'Vid2Skel': {
        "conversations": [{"from": "human", "value": None},
                          {"from": "gpt", "value": "<skeleton>"}],
        "videos": [],
        "skeletons": []
    },
    'SkelPred': {
        "conversations": [{"from": "human", "value": None},
                          {"from": "gpt", "value": "<skeleton>"}],
        "videos": [],
        "skeletons": []
    },
    'Text2Skel': {
        "conversations": [{"from": "human", "value": None},
                          {"from": "gpt", "value": "<skeleton>"}],
        "videos": [],
        "skeletons": [] # 只包含一个目标文件
    }
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint_data_type', type=str, required=True, choices=['joint3d_image_normed', 'joint3d_image_affined_normed'])
    parser.add_argument('--prompt_template', type=str, required=True, choices=['fixed', 'simple', 'bodypart_aware', 'BodypartAwareExplicit', 'joint_aware_explicit'])
    parser.add_argument('--task', type=str, required=True, choices=['Vid2Skel', 'SkelPred', 'Text2Skel'])
    parser.add_argument('--data_split', type=str, required=True)

    parser.add_argument('--resume_pth', type=str, required=True)
    parser.add_argument('--save_root', type=str, default='/home/wxs/LLaMA-Factory/data/_multimodal_data_byBrad/')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--nb_code', type=int, default=8192)
    parser.add_argument('--codebook_dim', type=int, default=3072)

    # Also defined in yaml. 如果在命令行中指定，则覆盖yaml中的配置
    parser.add_argument('--num_frames', type=int, default=None, help="Number of frames per sample.")
    parser.add_argument('--sample_stride', type=int, default=None)
    parser.add_argument('--data_stride', type=int, default=None)
    parser.add_argument('--data_mode', type=str, default=None)

    parser.add_argument('--load_data_file', type=str, default=None)
    parser.add_argument('--load_image_source_file', type=str, default=None)
    parser.add_argument('--load_bbox_file', type=str, default=None)
    parser.add_argument('--load_text_source_file', type=str, default=None)
    parser.add_argument('--return_extra', type=str, default=None)

    parser.add_argument('--normalize', type=str, default=None)
    parser.add_argument('--filter_invalid_images', type=str, default=None)
    parser.add_argument('--processed_image_shape', type=str, default=None)
    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--get_item_list', type=str, default=None)

    # VISION BACKBONE config. 如果在命令行中指定，则覆盖vision_config中的配置
    parser.add_argument('--hrnet_output_level', type=int, default=None, help="int or list. 0,1,2,3 分别对应输出 [B,32,H/4,W/4], [B,64,H/8,W/8], [B,128,H/16,W/16], [B,256,H/32,W/32] 的特征")
    parser.add_argument('--vision_guidance_ratio', type=float, default=None)

    args = parser.parse_args()

    if isinstance(args.return_extra, str):
        args.return_extra = ast.literal_eval(args.return_extra)
    if isinstance(args.processed_image_shape, str):
        args.processed_image_shape = ast.literal_eval(args.processed_image_shape)
    if isinstance(args.get_item_list, str):
        args.get_item_list = ast.literal_eval(args.get_item_list)
    if isinstance(args.hrnet_output_level, str):
        args.hrnet_output_level = ast.literal_eval(args.hrnet_output_level)
    if isinstance(args.filter_invalid_images, str):
        args.filter_invalid_images = args.filter_invalid_images.lower() == "true"

    vqvae_config.encoder.out_channels = args.codebook_dim
    vqvae_config.decoder.in_channels = args.codebook_dim
    vqvae_config.vq.nb_code = args.nb_code
    vqvae_config.vq.code_dim = args.codebook_dim
    vqvae_config.vq.is_train = False

    if args.hrnet_output_level is not None:
        vision_config.model.hybrid.hrnet_output_level = args.hrnet_output_level
    if args.vision_guidance_ratio is not None:
        vision_config.model.hybrid.vision_guidance_ratio = args.vision_guidance_ratio

    return args

def prepare_vqvae(ckpt_path, joint_data_type):
    vqvae = HYBRID_VQVAE(vqvae_config.encoder, vqvae_config.decoder, vqvae_config.vq, vision_config=vision_config, joint_data_type=joint_data_type)

    safetensors_path = os.path.join(ckpt_path, "model.safetensors")
    pytorch_bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        print(f"Loading model from {safetensors_path}")
        state_dict = load_safetensors(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_bin_path):
        print(f"Loading model from {pytorch_bin_path}")
        state_dict = torch.load(pytorch_bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in {ckpt_path}")

    vqvae.load_state_dict(state_dict, strict=True)
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False
    vqvae = vqvae.cuda()
    return vqvae

if __name__ == "__main__":
    args = get_args()
    vqvae = prepare_vqvae(args.resume_pth, args.joint_data_type)

    get_item_list=args.get_item_list
    if args.joint_data_type not in get_item_list:
        get_item_list = [args.joint_data_type, 
                         args.joint_data_type.replace('normed', 'scale'), 
                         args.joint_data_type.replace('normed', 'transl')] + get_item_list
    dataset = Multimodal_Mocap_Dataset(num_frames=args.num_frames, sample_stride=args.sample_stride, data_stride=args.data_stride,
                                                data_mode=args.data_mode,
                                                designated_split=args.data_split,
                                                load_data_file=args.load_data_file,
                                                load_image_source_file=args.load_image_source_file,
                                                load_bbox_file=args.load_bbox_file,
                                                load_text_source_file=args.load_text_source_file,
                                                return_extra=args.return_extra,
                                                # data preprocessing config
                                                normalize=args.normalize,  # isotropic (i.e., screen_coordinate_normalize), anisotropic
                                                # image config
                                                filter_invalid_images=args.filter_invalid_images,
                                                processed_image_shape=args.processed_image_shape,    # e.g., (192,256)
                                                backbone=args.backbone,
                                                # dataloader config
                                                get_item_list=get_item_list,
                                                batch_return_type='dict',
                                                )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=16,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    prompt_template_key = args.prompt_template
    task = args.task

   
    save_dir = osp.join(args.save_root, task, prompt_template_key, args.save_dir)
    source_data_save_dir = osp.join(save_dir, args.data_split)
    json_data_save_dir = osp.join(save_dir, f'{args.data_split}.jsonl')

    SOURCE_DATA_DICT = defaultdict(list)
    for batch in tqdm(dataloader):
        if isinstance(batch, tuple):
            batch_dict = {}
            assert len(batch) == len(dataloader.dataset.get_item_list)
            for element_id, element in enumerate(dataloader.dataset.get_item_list):
                batch_dict[element] = batch[element_id]
            batch = edict(batch_dict)

        joint3d_video = batch[args.joint_data_type].cuda()
        if args.vision_guidance_ratio > 0:
            video_rgb = torch.stack(batch['video_rgb']).cuda()  # [B,T,H,W,3]
        else:
            video_rgb = None

        with torch.no_grad():
            codebook_indices, quant_shape = vqvae.encode(joint3d_video=joint3d_video, video_rgb=video_rgb)
        joint3d_video = joint3d_video.cpu().numpy()
        codebook_indices = codebook_indices.cpu().numpy()   # (B, quant_t, 17). typically, quant_t = T//4
        quant_shape = np.array(quant_shape[1:])[None].repeat(quant_shape[0],0) # (B,3)

        SOURCE_DATA_DICT['skeleton_pose3d'].append(joint3d_video)
        SOURCE_DATA_DICT['skeleton_code'].append(codebook_indices)
        SOURCE_DATA_DICT['skeleton_quant_shape'].append(quant_shape)
        SOURCE_DATA_DICT['norm_scale'].append(batch[args.joint_data_type.replace('normed', 'scale')].cpu().numpy())
        SOURCE_DATA_DICT['norm_transl'].append(batch[args.joint_data_type.replace('normed', 'transl')].cpu().numpy())
        SOURCE_DATA_DICT['slice_id'].append(torch.stack(batch['slice_id']).cpu().numpy())
        if task == 'Vid2Skel':
            SOURCE_DATA_DICT['image_sources'].append(batch['image_sources'])
        if args.data_split == 'test':
            SOURCE_DATA_DICT['factor_2_5d'].append(torch.stack(batch['factor_2_5d']).cpu().numpy())

        if 'debugpy' in sys.modules:
            if len(SOURCE_DATA_DICT['skeleton_code']) >= 32:
                break

    for key in SOURCE_DATA_DICT:
        SOURCE_DATA_DICT[key] = np.concatenate(SOURCE_DATA_DICT[key], axis=0)
        print(f"{key}: shape={SOURCE_DATA_DICT[key].shape}")
    
    assert all([len(SOURCE_DATA_DICT[key])==len(SOURCE_DATA_DICT['skeleton_code']) for key in SOURCE_DATA_DICT])


    JSON_DATA_LIST = []
    for sample_id in tqdm(range(len(SOURCE_DATA_DICT['slice_id']))):
        for name_key, data_npy in SOURCE_DATA_DICT.items():
            if name_key in ['slice_id', 'image_sources']:
                continue
            save_path = os.path.join(source_data_save_dir, name_key)
            if not os.path.exists(save_path): 
                os.makedirs(save_path)

            name_suffix = f"slice{int(SOURCE_DATA_DICT['slice_id'][sample_id][0])}_{int(SOURCE_DATA_DICT['slice_id'][sample_id][-1]+1)}"
            save_file = os.path.join(save_path, f"h36m_{sample_id:06d}_{name_suffix}.npy")
            np.save(save_file, data_npy[sample_id])

            if name_key == 'skeleton_code':
                skeleton_code_save_file = save_file

        task_item = edict.EasyDict(TASK_TEMPLATE[task])
        chosen_prompt = random.choice(PROMPT_TEMPLATES[task][prompt_template_key])
        task_item.conversations[0]["value"] = chosen_prompt
        task_item.skeletons = [skeleton_code_save_file]
        if task == 'Vid2Skel':
            task_item.videos = [SOURCE_DATA_DICT['image_sources'][sample_id].tolist()]

        JSON_DATA_LIST.append(task_item)

    print(f"Saving to {json_data_save_dir}. Total {len(JSON_DATA_LIST)} samples.")
    with open(json_data_save_dir, 'w') as f:
        for item in JSON_DATA_LIST:
            f.write(json.dumps(item) + '\n')
