import os
import os.path as osp
import torch
import numpy as np
import joblib
import easydict
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
sys.path.append('/home/wxs/MTVCrafter/config/')
from vision_backbone import config as vision_config
from vqvae import vqvae_config
sys.path.remove('/home/wxs/MTVCrafter/config/')


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
        'bodypart_aware_explicit': [
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
        'bodypart_aware_explicit': [
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
        'bodypart_aware_explicit': [
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
    parser.add_argument('--prompt_template', type=str, required=True, choices=['fixed', 'simple', 'bodypart_aware', 'bodypart_aware_explicit', 'joint_aware_explicit'])
    parser.add_argument('--task', type=str, required=True, choices=['Vid2Skel', 'SkelPred', 'Text2Skel'])
    parser.add_argument('--data_split', type=str, required=True)

    parser.add_argument('--resume_pth', type=str, required=True)
    parser.add_argument('--save_root', type=str, default='/home/wxs/LLaMA-Factory/data/_multimodal_data_byBrad/')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--nb_code', type=int, default=8192)
    parser.add_argument('--codebook_dim', type=int, default=3072)

    parser.add_argument('--vqvae_type', type=str, default='base')

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
    parser.add_argument('--filter_invalid_images', type=bool, default=None)
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

    vqvae_config.encoder.out_channels = args.codebook_dim
    vqvae_config.vq.nb_code = args.nb_code
    vqvae_config.vq.code_dim = args.codebook_dim
    vqvae_config.decoder.in_channels = args.codebook_dim

    return args

def prepare_vqvae(vqvae_config, ckpt_path):
    encoder = Encoder(**vqvae_config.encoder)
    vq = VectorQuantizer(**vqvae_config.vq, is_train=False)
    decoder = Decoder(**vqvae_config.decoder)
    vqvae = SKEL_VQVAE(encoder, decoder, vq)

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
    vqvae = prepare_vqvae(vqvae_config, args.resume_pth)

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
                                                get_item_list=args.get_item_list,
                                                batch_return_type='dict',
                                                )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    prompt_template_key = args.prompt_template
    task = args.task

   
    save_dir = osp.join(args.save_root, task, prompt_template_key, args.save_dir)
    source_data_save_dir = osp.join(save_dir, args.data_split)
    json_data_save_dir = osp.join(save_dir, f'{args.data_split}.jsonl')

    source_data_dict = defaultdict(list)
    


    


def img_to_skel():

    POSES = []
    CODEBOOK_INDICES = []
    QUANT_SHAPES = []
    IMAGES = []
    SLICED_INDICES = []
    for batch in tqdm(img2skel_dataloader):
        pose_seq, img_src, _, slice_indices = batch
        # pose_seq: (B,T,17,3)
        # img_src: B-length list of T-length lists. img_src[b][t] is a str
        pose_seq = pose_seq.cuda()
        with torch.no_grad():
            codebook_indices, quant_shape = skeleton_processor.encode(pose_seq)
        codebook_indices = codebook_indices.cpu().numpy()   # (B, quant_t, 17). typically, quant_t = T//4
        quant_shape = np.array(quant_shape[1:])[None].repeat(quant_shape[0],0) # (B,3)

        POSES.append(pose_seq.cpu().numpy())
        CODEBOOK_INDICES.append(codebook_indices)
        QUANT_SHAPES.append(quant_shape)
        IMAGES = IMAGES + img_src
        SLICED_INDICES = SLICED_INDICES + slice_indices
    POSES = np.concatenate(POSES, axis=0)                      # (N, T, 17, 3)
    CODEBOOK_INDICES = np.concatenate(CODEBOOK_INDICES, axis=0)  # (N, quant_t, 17)
    QUANT_SHAPES = np.concatenate(QUANT_SHAPES, axis=0)          # (N, 3)
    assert CODEBOOK_INDICES.shape[0] == QUANT_SHAPES.shape[0] == len(IMAGES)

    jsonl_data = []

    for sample_id in tqdm(range(len(IMAGES))):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pose_save_path = f"{save_path}/skeleton_pose3d"
        pose_save_file = os.path.join(pose_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(pose_save_path): 
            os.makedirs(pose_save_path)
        pose = POSES[sample_id]                         # (T, 17, 3)
        np.save(pose_save_file, pose)

        codebook_index_save_path = f"{save_path}/skeleton_code"
        codebook_index_save_file = os.path.join(codebook_index_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(codebook_index_save_path): 
            os.makedirs(codebook_index_save_path)
        codebook_index = CODEBOOK_INDICES[sample_id]   # (quant_t, 17)
        np.save(codebook_index_save_file, codebook_index)

        quant_shape_save_path = f"{save_path}/skeleton_quant_shape"
        quant_shape_save_file = os.path.join(quant_shape_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(quant_shape_save_path): 
            os.makedirs(quant_shape_save_path)
        quant_shape = QUANT_SHAPES[sample_id]           # (3)
        np.save(quant_shape_save_file, quant_shape)

        source_slice_id_save_path = f"{save_path}/source_slice_id"
        source_slice_id_save_file = os.path.join(source_slice_id_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(source_slice_id_save_path): 
            os.makedirs(source_slice_id_save_path)
        source_slice_id = SLICED_INDICES[sample_id]
        np.save(source_slice_id_save_file, source_slice_id)

        task_item = easydict.EasyDict(TASK_TEMPLATE['img_to_skel'])
        chosen_prompt = random.choice(PROMPT_TEMPLATES['img_to_skel'][prompt_template_key])
        task_item.conversations[0]["value"] = chosen_prompt
        task_item.videos = [IMAGES[sample_id]]
        task_item.skeletons = [codebook_index_save_file]

        jsonl_data.append(task_item)

    if not os.path.exists(os.path.dirname(jsonl_save_file)):
        os.makedirs(os.path.dirname(jsonl_save_file))
    with open(jsonl_save_file, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')


def skel_pred():
    num_frames = 16
    sample_stride = 1
    data_stride = 16
    designated_split = 'test'

    # prompt_template_key = 'fixed'
    # prompt_template_key = 'simple'
    # prompt_template_key = 'bodypart_aware'
    prompt_template_key = 'bodypart_aware_explicit'

    load_data_file = "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl"
    # load_data_file = "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl,/data2/wxs/DATASETS/AMASS_ByBradley/,/data2/wxs/DATASETS/PW3D_ByBradley/all_data.pkl"

    # save_path = f'/home/wxs/LLaMA-Factory/data/source_data_byBrad/skel_pred/ams_h36m_3dpw/f{num_frames}s{sample_stride}d{data_stride}{"" if prompt_template_key=="simple" else "_"+prompt_template_key}/{designated_split}'
    # jsonl_save_file = f'/home/wxs/LLaMA-Factory/data/custom_dataset_byBrad/skel_pred/ams_h36m_3dpw/f{num_frames}s{sample_stride}d{data_stride}{"" if prompt_template_key=="simple" else "_"+prompt_template_key}/{designated_split}.jsonl'
    save_path = f'/home/wxs/LLaMA-Factory/data/source_data_byBrad/skel_pred/f{num_frames}s{sample_stride}d{data_stride}{"" if prompt_template_key=="simple" else "_"+prompt_template_key}/{designated_split}'
    jsonl_save_file = f'/home/wxs/LLaMA-Factory/data/custom_dataset_byBrad/skel_pred/f{num_frames}s{sample_stride}d{data_stride}{"" if prompt_template_key=="simple" else "_"+prompt_template_key}/{designated_split}.jsonl'

    load_image_source_file = ""
    load_text_source_file = ""

    skeleton_processor = prepare_vqvae(mode='joint3d', sample_stride=sample_stride)
    skel_dataset = SkeletonDataset(num_frames=num_frames * 2, sample_stride=sample_stride, data_stride=data_stride, data_mode='joint3d', designated_split=designated_split,
                                       load_data_file=load_data_file, load_image_source_file=load_image_source_file, load_text_source_file=load_text_source_file,
                                       return_extra=[[]],
                                       )
    skel_dataloader = torch.utils.data.DataLoader(skel_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    POSES = {'history': [], 'future': []}
    CODEBOOK_INDICES = {'history': [], 'future': []}
    QUANT_SHAPES = {'history': [], 'future': []}
    SLICED_INDICES = []
    for batch in tqdm(skel_dataloader):
        pose_seq, _, _, slice_indices = batch
        # pose_seq: (B,2T,17,3)
        # img_src: B-length list of T-length lists. img_src[b][t] is a str
        pose_seq = pose_seq.cuda()
        pose_seq_history, pose_seq_future = pose_seq.chunk(2, dim=1)  # (B,T,17,3), (B,T,17,3)

        with torch.no_grad():
            codebook_indices_history, quant_shape_history = skeleton_processor.encode(pose_seq_history)
            codebook_indices_future, quant_shape_future = skeleton_processor.encode(pose_seq_future)

        codebook_indices_history = codebook_indices_history.cpu().numpy()   # (B, quant_t, 17). typically, quant_t = T//4
        quant_shape_history = np.array(quant_shape_history[1:])[None].repeat(quant_shape_history[0],0) # (B,3)
        codebook_indices_future = codebook_indices_future.cpu().numpy()   # (B, quant_t, 17). typically, quant_t = T//4
        quant_shape_future = np.array(quant_shape_future[1:])[None].repeat(quant_shape_future[0],0) # (B,3)

        POSES['history'].append(pose_seq_history.cpu().numpy())
        POSES['future'].append(pose_seq_future.cpu().numpy())
        CODEBOOK_INDICES['history'].append(codebook_indices_history)
        CODEBOOK_INDICES['future'].append(codebook_indices_future)
        QUANT_SHAPES['history'].append(quant_shape_history)
        QUANT_SHAPES['future'].append(quant_shape_future)

        SLICED_INDICES = SLICED_INDICES + slice_indices

    POSES = {k: np.concatenate(v, axis=0) for k, v in POSES.items()}
    CODEBOOK_INDICES = {k: np.concatenate(v, axis=0) for k, v in CODEBOOK_INDICES.items()}
    QUANT_SHAPES = {k: np.concatenate(v, axis=0) for k, v in QUANT_SHAPES.items()}
    assert CODEBOOK_INDICES['history'].shape[0] == QUANT_SHAPES['history'].shape[0]
    assert CODEBOOK_INDICES['future'].shape[0] == QUANT_SHAPES['future'].shape[0]

    jsonl_data = []

    for sample_id in tqdm(range(CODEBOOK_INDICES['history'].shape[0])):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        history_pose_save_path = f"{save_path}/history/skeleton_pose3d"
        history_pose_save_file = os.path.join(history_pose_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(history_pose_save_path): 
            os.makedirs(history_pose_save_path)
        future_pose_save_path = f"{save_path}/future/skeleton_pose3d"
        future_pose_save_file = os.path.join(future_pose_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(future_pose_save_path): 
            os.makedirs(future_pose_save_path)
        pose_his = POSES['history'][sample_id]                         # (T, 17, 3)
        pose_fut = POSES['future'][sample_id]                         # (T, 17, 3)
        np.save(history_pose_save_file, pose_his)
        np.save(future_pose_save_file, pose_fut)

        history_codebook_index_save_path = f"{save_path}/history/skeleton_code"
        history_codebook_index_save_file = os.path.join(history_codebook_index_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(history_codebook_index_save_path): 
            os.makedirs(history_codebook_index_save_path)
        future_codebook_index_save_path = f"{save_path}/future/skeleton_code"
        future_codebook_index_save_file = os.path.join(future_codebook_index_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(future_codebook_index_save_path): 
            os.makedirs(future_codebook_index_save_path)
        codebook_index_his = CODEBOOK_INDICES['history'][sample_id]   # (quant_t, 17)
        codebook_index_fut = CODEBOOK_INDICES['future'][sample_id]   # (quant_t, 17)
        np.save(history_codebook_index_save_file, codebook_index_his)
        np.save(future_codebook_index_save_file, codebook_index_fut)

        history_quant_shape_save_path = f"{save_path}/history/skeleton_quant_shape"
        history_quant_shape_save_file = os.path.join(history_quant_shape_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(history_quant_shape_save_path):
            os.makedirs(history_quant_shape_save_path)
        future_quant_shape_save_path = f"{save_path}/future/skeleton_quant_shape"
        future_quant_shape_save_file = os.path.join(future_quant_shape_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(future_quant_shape_save_path):
            os.makedirs(future_quant_shape_save_path)
        quant_shape_his = QUANT_SHAPES['history'][sample_id]           # (3)
        quant_shape_fut = QUANT_SHAPES['future'][sample_id]           # (3)
        np.save(history_quant_shape_save_file, quant_shape_his)
        np.save(future_quant_shape_save_file, quant_shape_fut)

        source_slice_id_save_path = f"{save_path}/source_slice_id"
        source_slice_id_save_file = os.path.join(source_slice_id_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(source_slice_id_save_path): 
            os.makedirs(source_slice_id_save_path)
        source_slice_id = SLICED_INDICES[sample_id]
        np.save(source_slice_id_save_file, source_slice_id)

        task_item = easydict.EasyDict(TASK_TEMPLATE['skel_pred'])
        chosen_prompt = random.choice(PROMPT_TEMPLATES['skel_pred'][prompt_template_key])
        task_item.conversations[0]["value"] = chosen_prompt
        task_item.skeletons = [history_codebook_index_save_file, future_codebook_index_save_file]

        jsonl_data.append(task_item)

    if not os.path.exists(os.path.dirname(jsonl_save_file)):
        os.makedirs(os.path.dirname(jsonl_save_file))
    with open(jsonl_save_file, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')


def text_to_skel():
    num_frames = 64
    sample_stride = 2
    data_stride = 64
    designated_split = 'test'

    prompt_template_key = 'fixed'
    # prompt_template_key = 'simple'
    # prompt_template_key = 'bodypart_aware'

    save_path = f'/home/wxs/LLaMA-Factory/data/source_data_byBrad/text_to_skel/f{num_frames}s{sample_stride}d{data_stride}{"" if prompt_template_key=="simple" else "_"+prompt_template_key}/{designated_split}'
    jsonl_save_file = f'/home/wxs/LLaMA-Factory/data/custom_dataset_byBrad/text_to_skel/f{num_frames}s{sample_stride}d{data_stride}{"" if prompt_template_key=="simple" else "_"+prompt_template_key}/{designated_split}.jsonl'

    load_data_file = "/data2/wxs/DATASETS/AMASS_ByBradley/"
    load_image_source_file = ""
    load_text_source_file = "/data2/wxs/DATASETS/AMASS_ByBradley/text_map.pkl"

    skeleton_processor = prepare_vqvae(mode='joint3d', sample_stride=sample_stride)
    text2skel_dataset = SkeletonDataset(num_frames=num_frames, sample_stride=sample_stride, data_stride=data_stride, 
                                        data_mode='joint3d', designated_split=designated_split,
                                        load_data_file=load_data_file, load_image_source_file=load_image_source_file, load_text_source_file=load_text_source_file,
                                        return_extra=[['text']],
                                       )
    img2skel_dataloader = torch.utils.data.DataLoader(text2skel_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    POSES = []
    CODEBOOK_INDICES = []
    QUANT_SHAPES = []
    CAPTIONS = []
    for batch in tqdm(img2skel_dataloader):
        raise NotImplementedError("need to modify SkeletonDataset to return slice_id for each sample")
        pose_seq, _, caption = batch
        # pose_seq: (B,T,17,3)
        # img_src: B-length list of T-length lists. img_src[b][t] is a str
        pose_seq = pose_seq.cuda()
        with torch.no_grad():
            codebook_indices, quant_shape = skeleton_processor.encode(pose_seq)
        codebook_indices = codebook_indices.cpu().numpy()   # (B, quant_t, 17). typically, quant_t = T//4
        quant_shape = np.array(quant_shape[1:])[None].repeat(quant_shape[0],0) # (B,3)

        POSES.append(pose_seq.cpu().numpy())
        CODEBOOK_INDICES.append(codebook_indices)
        QUANT_SHAPES.append(quant_shape)
        CAPTIONS = CAPTIONS + caption
    POSES = np.concatenate(POSES, axis=0)                      # (N, T, 17, 3)
    CODEBOOK_INDICES = np.concatenate(CODEBOOK_INDICES, axis=0)  # (N, quant_t, 17)
    QUANT_SHAPES = np.concatenate(QUANT_SHAPES, axis=0)          # (N, 3)
    assert CODEBOOK_INDICES.shape[0] == QUANT_SHAPES.shape[0] == len(CAPTIONS)

    jsonl_data = []

    for sample_id in tqdm(range(len(CAPTIONS))):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pose_save_path = f"{save_path}/skeleton_pose3d"
        pose_save_file = os.path.join(pose_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(pose_save_path): 
            os.makedirs(pose_save_path)
        pose = POSES[sample_id]                         # (T, 17, 3)
        np.save(pose_save_file, pose)

        codebook_index_save_path = f"{save_path}/skeleton_code"
        codebook_index_save_file = os.path.join(codebook_index_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(codebook_index_save_path): 
            os.makedirs(codebook_index_save_path)
        codebook_index = CODEBOOK_INDICES[sample_id]   # (quant_t, 17)
        np.save(codebook_index_save_file, codebook_index)

        quant_shape_save_path = f"{save_path}/skeleton_quant_shape"
        quant_shape_save_file = os.path.join(quant_shape_save_path, f"h36m_{sample_id:06d}.npy")
        if not os.path.exists(quant_shape_save_path): 
            os.makedirs(quant_shape_save_path)
        quant_shape = QUANT_SHAPES[sample_id]           # (3)
        np.save(quant_shape_save_file, quant_shape)

        task_item = easydict.EasyDict(TASK_TEMPLATE['text_to_skel'])
        chosen_prompt = random.choice(PROMPT_TEMPLATES['text_to_skel'][prompt_template_key])
        prompt_with_caption = chosen_prompt.replace("<text_description>", CAPTIONS[sample_id])
        task_item.conversations[0]["value"] = prompt_with_caption
        task_item.skeletons = [codebook_index_save_file]

        jsonl_data.append(task_item)

    if not os.path.exists(os.path.dirname(jsonl_save_file)):
        os.makedirs(os.path.dirname(jsonl_save_file))
    with open(jsonl_save_file, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')


def img_to_text():
    load_data_file = "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl"
    source_list = joblib.load(load_data_file)['train']['source']
    split_id = split_clips(source_list, n_frames=num_frames, data_stride=num_frames)

    load_image_source_file = "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/images_source.pkl"
    img_list = joblib.load(load_image_source_file)['train']
    # img_list = [img_path.replace('images_fps50', f'images_fps50_cropped_192x256') for img_path in img_list]
    img_list = np.array(img_list)

    video_list = img_list[split_id].tolist()  # (N, num_frames)

    generate_pseudo_labels(video_list)

def generate_pseudo_labels(image_sequences: list[list[str]]) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
    """
    使用原始的 Qwen2.5-VL-Instruct 模型为每个图片序列生成描述。
    """
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    torch_dtype = torch.bfloat16
    device = "cuda"

    print(f"Loading captioning model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    prompt_template = "Describe the motion of the person in this video. Focus on the motion of the whole body and the movement of body parts and joints over time. Ignore the direction towards which the person face or the facial expression or the background or other objects. Be concise and specific."
    
    captions = []
    for i, image_paths in enumerate(tqdm(image_sequences, desc="Generating Captions")):
        # 准备模型的输入
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_template}]}]
        
        # 加载图片序列
        video_frames = [Image.open(p).convert("RGB") for p in image_paths]
        messages[0]["content"].insert(1, {"type": "image", "content": video_frames})

        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_tensors="pt",
            return_dict=True
        ).to(device)

        # 生成描述
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        
        # 解码并清理输出
        response = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # 从 "assistant\n" 之后开始截取，并移除特殊 token
        try:
            clean_caption = response.split("assistant\n")[1]
            clean_caption = clean_caption.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            captions.append(clean_caption)
        except IndexError:
            print(f"Warning: Could not parse caption for sample {i}. Using empty string.")
            captions.append("")

    del model
    torch.cuda.empty_cache()
    return captions
   
class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=16, sample_stride=1, data_stride=16, data_mode="joint3d", designated_split='train',
                 load_data_file="", load_image_source_file="", load_text_source_file="",
                 return_extra=[['image'], ['text']],                 
                 # e.g.,
                 # lode_data_file='<h36m_path>,<amass_path>'
                 # load_image_source_file='<h36m_img_path>,'
                 # load_text_source_file=',<amass_text_path>'
                 # return_extra=[['image'], ['text']]
                 use_cropped_image=True, image_shape='192x256',
                 ):
        assert len(load_data_file.split(',')) == len(load_image_source_file.split(',')) == len(return_extra)

        self.num_frames = num_frames

        data_dict = {}
        data_list = []
        for dt_file, img_src_file, txt_src_file, extra_modality_list in zip(load_data_file.split(','), load_image_source_file.split(','), load_text_source_file.split(','), return_extra):
            datareader_config_unsplit = {'dt_file': dt_file,}
            datareader_config_split = {'chunk_len': num_frames,
                                       'sample_stride': sample_stride, 
                                       'data_stride': data_stride,
                                       'read_confidence': False}
            datareader_config = {**datareader_config_unsplit, **datareader_config_split}
            datareader = DataReaderMesh(**datareader_config)        
            unsplit_data = DataReaderMesh.load_dataset_static(**datareader_config_unsplit)   # '/data2/wxs/DATASETS/AMASS_ByBradley'
            data_size_ori = len(unsplit_data[designated_split]['source'])
            datareader.dt_dataset = unsplit_data
            read_func = datareader.read_2d if data_mode == "joint2d" else datareader.read_3d_image
            data_npy = read_func(designated_split=designated_split)     # (N,17,3)
            data_sources = datareader.read_source(designated_split=designated_split)    # sampled_stride applied within read_source

            data_dict[dt_file] = {'poses': data_npy, 'sources': data_sources}


            if 'text' in extra_modality_list:
                assert 'image' not in extra_modality_list, "image and text cannot be requested at the same time yet."
                if 'AMASS' not in txt_src_file:
                    raise NotImplementedError("text modality only implemented for AMASS yet.")
                text_map = joblib.load(txt_src_file)[designated_split]
                video_map = joblib.load(txt_src_file.replace('text_map', 'video_map'))[designated_split]

                # Get split_id
                split_id = []
                captions_list = []
                for video_id, clip_dict in text_map.items():
                    for clip_key, captions in clip_dict.items():
                        original_global_indices = video_map[video_id][clip_key]  # range(0,180)
                        original_global_indices = np.array(original_global_indices)

                        start_index = original_global_indices[0]
                        if start_index % sample_stride != 0:
                            start_index = start_index + (sample_stride - start_index % sample_stride)
                        
                        # 3. 从调整后的起点开始，以 sample_stride 为步长，选出所有有效的原始索引
                        sampled_original_indices = np.arange(start_index, original_global_indices[-1] + 1, sample_stride)

                        # 4. 【核心步骤】将下采样后的原始索引，映射到下采样后数据集的新索引
                        #    这通过整数除法完成
                        new_global_indices = sampled_original_indices // sample_stride

                        # 5. 由于你的 Dataset __getitem__ 需要一个 slice，我们用 new_global_indices 创建它
                        #    注意：这里我们假设 new_global_indices 是连续的，如果不是，需要更复杂的处理
                        #    但根据arange的用法，它一定是连续的。
                        if len(new_global_indices) >= num_frames:
                            start, end = new_global_indices[0], new_global_indices[-1] + 1
                            new_slice = slice(start, end)

                            split_id.extend([new_slice]*len(captions))
                            captions_list.extend(captions)

                data_list.extend(zip([dt_file]*len(split_id), split_id, captions_list))

            else:
                if 'image' in extra_modality_list:
                    assert 'text' not in extra_modality_list, "image and text cannot be requested at the same time yet."                    
                    img_list = joblib.load(img_src_file)[designated_split]
                    img_list = img_list[::sample_stride]
                    valid_img_indices = []
                    for frame_id, img_path in enumerate(img_list):
                        if img_path is None:
                            continue
                        valid_img_indices.append(frame_id)
                        if use_cropped_image:
                            img_list[frame_id] = img_path.replace('images_fps50', f'images_fps50_cropped_{image_shape}')
                    img_list = np.array(img_list)[valid_img_indices]
                    data_dict[dt_file]['img_src'] = img_list
                else:
                    valid_img_indices = slice(None, None)
                data_npy = data_npy[valid_img_indices]
                
                if 'image' in extra_modality_list:
                    assert len(img_list) == data_npy.shape[0]

                datareader.dt_dataset[designated_split]['source'] = np.array(datareader.dt_dataset[designated_split]['source'])[valid_img_indices].tolist()
                
                # Get split_id
                split_id = datareader.get_split_id(designated_split=designated_split)   # 这里是用 unsplit_data 中的 'source' 来划分 split_id, 所以也要利用 valid_indices 作修改

                data_list.extend(zip([dt_file]*len(split_id), split_id, [None]*len(split_id)))


        self.data_dict = data_dict
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        dt_file, slice_id, caption = self.data_list[idx]
        # caption could be None if it's a sample from pose-image sets (e.g., H36M)
        poses = self.data_dict[dt_file]['poses'][slice_id]

        if caption is not None:
            idx = random.randint(0, poses.shape[0] - self.num_frames)
            poses = poses[idx:idx + self.num_frames]

        if 'img_src' in self.data_dict[dt_file]:
            img_src = self.data_dict[dt_file]['img_src'][slice_id].tolist()
        else:
            img_src = []

        return torch.from_numpy(poses).float(), img_src, caption, slice_id

def prepare_vqvae(mode='joint3d', sample_stride=1):
    encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1])
    vq = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)
    decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
    skeleton_processor = SkeletonProcessor(encoder, decoder, vq)

    if mode == 'joint3d' and sample_stride == 1:
        ckpt_path = "/home/wxs/LLaMA-Factory/src/llamafactory/extras_byBrad/vqvae_experiment/all_datasets_j3d/models/checkpoint_epoch_113_step_500000/model.safetensors"
    elif mode == 'joint3d' and sample_stride == 2:
        ckpt_path = "/home/wxs/LLaMA-Factory/src/llamafactory/extras_byBrad/vqvae_experiment/all_datasets_j3d_f64s2/models/checkpoint_epoch_148_step_240000/model.safetensors"
    else:
        raise NotImplementedError
    
    state_dict = load_file(ckpt_path, device="cpu")
    skeleton_processor.load_state_dict(state_dict)
    skeleton_processor.eval()
    for param in skeleton_processor.parameters():
        param.requires_grad = False
    skeleton_processor = skeleton_processor.cuda()
    return skeleton_processor


if __name__ == "__main__":
    # skel_pred()
    # text_to_skel()
    img_to_skel()