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
from time import time


from llamafactory.extras.constants import PROMPT_PLACEHOLDER, RESPONSE_PLACEHOLDER


import sys
sys.path.append("../ContextAwarePoseFormer_Private/H36M-Toolbox/")
from multimodal_h36m_dataset_byBradley import Multimodal_Mocap_Dataset, DATA_ROOT_PATH
sys.path.remove("../ContextAwarePoseFormer_Private/H36M-Toolbox/")
# sys.path.append('../Skeleton-in-Context-tpami/')
# from lib.utils.viz_skel_seq import viz_skel_seq_anim
# sys.path.remove('../Skeleton-in-Context-tpami/')
sys.path.append('../MTVCrafter/')
from config.vision_backbone import config as vision_config
from config.vqvae import vqvae_config
from models import HYBRID_VQVAE
sys.path.remove('../MTVCrafter/')


assert 'LLaMA-Factory' in osp.abspath(__file__)
HOME_ROOT_PATH = osp.abspath(__file__).split('LLaMA-Factory')[0]
sys.path.append(osp.join(HOME_ROOT_PATH, 'LLaMA-Factory', "_llamafactory_skeleton_byBrad"))
from data_utils.templates import PROMPT_TEMPLATES, TASK_TEMPLATE
from data_utils.utils import data_prefetcher
sys.path.remove(osp.join(HOME_ROOT_PATH, 'LLaMA-Factory', "_llamafactory_skeleton_byBrad"))



IS_DEBUG = 'debugpy' in sys.modules


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default=osp.join(HOME_ROOT_PATH, 'LLaMA-Factory/_llamafactory_skeleton_byBrad/data'))
    parser.add_argument('--save_subdir_raw', type=str, required=True)
    parser.add_argument('--save_subdir_vqvae', type=str, required=True)
    parser.add_argument('--save_subdir_jsonl', type=str, required=True)
    
    # dataset config #######################################################
    parser.add_argument('--data_split', type=str, required=True)
    parser.add_argument('--num_frames', type=int, default=None, help="Number of frames per sample.")
    parser.add_argument('--sample_stride', type=int, default=None)
    parser.add_argument('--data_stride', type=int, default=None)
    parser.add_argument('--return_extra', type=str, default=None)
    parser.add_argument('--get_item_list', type=str, default=None)
    parser.add_argument('--save_item_list', type=str, default=None)
    parser.add_argument('--load_data_file', type=str, default=None)
    parser.add_argument('--load_image_source_file', type=str, default=None)
    parser.add_argument('--load_bbox_file', type=str, default=None)
    parser.add_argument('--load_text_source_file', type=str, default=None)

    parser.add_argument('--normalize', type=str, default=None)
    parser.add_argument('--filter_invalid_images', type=str, default=None)
    parser.add_argument('--processed_image_shape', type=str, default=None)
    parser.add_argument('--backbone', type=str, default=None)

    # useless
    parser.add_argument('--data_mode', type=str, default='joint3d')

    # vqvae config #######################################################
    parser.add_argument('--joint_data_type', type=str, required=True)
    parser.add_argument('--resume_pth', type=str, required=True)
    parser.add_argument('--nb_code', type=int, default=8192)
    parser.add_argument('--codebook_dim', type=int, default=3072)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hrnet_output_level', type=int, default=None, help="int or list. 0,1,2,3 分别对应输出 [B,32,H/4,W/4], [B,64,H/8,W/8], [B,128,H/16,W/16], [B,256,H/32,W/32] 的特征")
    parser.add_argument('--vision_guidance_ratio', type=float, default=None)
    parser.add_argument('--downsample_time', type=str, default=None)
    parser.add_argument('--frame_upsample_rate', type=str, default=None)


    # jsonl config #######################################################
    # parser.add_argument('--prompt_template', type=str, required=True, choices=['fixed', 'simple', 'bodypart_aware', 'BodypartAwareExplicit', 'JointAwareExplicit'])
    parser.add_argument('--prompt_template', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, choices=['Vid2Skel', 'SkelPred', 'Text2Skel'])


    args = parser.parse_args()

    if isinstance(args.return_extra, str):
        args.return_extra = ast.literal_eval(args.return_extra)
    if isinstance(args.processed_image_shape, str):
        args.processed_image_shape = ast.literal_eval(args.processed_image_shape)
    if isinstance(args.get_item_list, str):
        args.get_item_list = ast.literal_eval(args.get_item_list)
    if isinstance(args.save_item_list, str):
        args.save_item_list = ast.literal_eval(args.save_item_list)
    if isinstance(args.hrnet_output_level, str):
        args.hrnet_output_level = ast.literal_eval(args.hrnet_output_level)
    if isinstance(args.filter_invalid_images, str):
        args.filter_invalid_images = args.filter_invalid_images.lower() == "true"

    vqvae_config.encoder.out_channels = args.codebook_dim
    vqvae_config.decoder.in_channels = args.codebook_dim
    vqvae_config.vq.nb_code = args.nb_code
    vqvae_config.vq.code_dim = args.codebook_dim
    vqvae_config.vq.is_train = False

    if isinstance(args.downsample_time, str):
        args.downsample_time = ast.literal_eval(args.downsample_time)
    if isinstance(args.frame_upsample_rate, str):
        args.frame_upsample_rate = ast.literal_eval(args.frame_upsample_rate)
    if args.downsample_time is not None:
        vqvae_config.encoder.downsample_time = args.downsample_time
    if args.frame_upsample_rate is not None:
        vqvae_config.decoder.frame_upsample_rate = args.frame_upsample_rate

    if args.hrnet_output_level is not None:
        vision_config.model.hybrid.hrnet_output_level = args.hrnet_output_level
    if args.vision_guidance_ratio is not None:
        vision_config.model.hybrid.vision_guidance_ratio = args.vision_guidance_ratio

    return args

def prepare_vqvae(ckpt_path, joint_data_type):
    vqvae = HYBRID_VQVAE(vqvae_config.encoder, vqvae_config.decoder, vqvae_config.vq, vision_config=vision_config, joint_data_type=joint_data_type)

    vqvae.load_model_weights(ckpt_path)

    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False
    vqvae = vqvae.cuda()
    return vqvae


def load_dataset(args, dataset_args_file, slice_indices_file):
    get_item_list=args.get_item_list
    if args.joint_data_type not in get_item_list:
        try:
            get_item_list = [args.joint_data_type, 
                            args.joint_data_type.replace('normed', 'scale'), 
                            args.joint_data_type.replace('normed', 'transl')] + get_item_list
        except Exception as e:
            get_item_list = [args.joint_data_type] + get_item_list



    dataset_args = dict(
        designated_split=args.data_split,
        num_frames=args.num_frames, sample_stride=args.sample_stride, data_stride=args.data_stride,
        return_extra=args.return_extra,
        get_item_list=args.get_item_list,
        load_data_file=args.load_data_file,
        load_image_source_file=args.load_image_source_file,
        load_bbox_file=args.load_bbox_file,
        load_text_source_file=args.load_text_source_file,
        normalize=args.normalize,  # isotropic (i.e., screen_coordinate_normalize), anisotropic
        filter_invalid_images=args.filter_invalid_images,
        processed_image_shape=args.processed_image_shape,    # e.g., (192,256)
        backbone=args.backbone,
        # data_mode=args.data_mode,
    )

    # TODO
    dataset_args_str = ' - '.join([f"{k}={v}" for k, v in sorted(dataset_args.items())])
    dataset_args_hash = str(abs(hash(dataset_args_str)) % (10 ** 8))    
    
    if not osp.exists(dataset_args_file) and not IS_DEBUG:
        with open(dataset_args_file, 'w') as f:
            json.dump(dataset_args, f, indent=4)


    dataset = Multimodal_Mocap_Dataset(
        max_samples=512 if IS_DEBUG else 999999999,
        **dataset_args
    )

    SLICE_INDICES_LIST = []
    for data_id in tqdm(range(len(dataset))):
        _, slice_indices, _, _ = dataset.data_list[data_id] # avoid using __getitem__ as reading images can be very slow
        slice_indices_dict = {'start_id': slice_indices[0], 'end_id': slice_indices[-1]+1}
        SLICE_INDICES_LIST.append(slice_indices_dict)
    
    if not osp.exists(slice_indices_file) and not IS_DEBUG:
        with open(slice_indices_file, 'w') as f:
            for item in SLICE_INDICES_LIST:
                f.write(json.dumps(item) + '\n')

    return dataset


def get_skeleton_code(args, dataset, task):
    vqvae = prepare_vqvae(args.resume_pth, args.joint_data_type)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=16,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    with torch.no_grad():
        prefetcher = data_prefetcher(loader=dataloader,
                                     device=torch.device(0)
                                     )
        batch = prefetcher.next()
        data_len = len(dataloader)
        pbar = tqdm(total=data_len)

        num_iter = 0
        SOURCE_DATA_DICT = defaultdict(list)
        while batch is not None:
            if isinstance(batch, tuple):
                batch_dict = {}
                assert len(batch) == len(dataloader.dataset.get_item_list)
                for element_id, element in enumerate(dataloader.dataset.get_item_list):
                    batch_dict[element] = batch[element_id]
                batch = edict(batch_dict)

            joint3d_video = batch[args.joint_data_type]
            if args.vision_guidance_ratio > 0:
                video_rgb = batch['video_rgb'].cuda()  # [B,T,H,W,3]
            else:
                video_rgb = None

            codebook_indices, quant_shape = vqvae.encode(joint3d_video=joint3d_video, video_rgb=video_rgb)
            codebook_indices = codebook_indices.cpu().numpy()   # (B, quant_t, 17). typically, quant_t = T//4
            quant_shape = np.array(quant_shape[1:])[None].repeat(quant_shape[0],0) # (B,3)

            SOURCE_DATA_DICT[f'{args.joint_data_type}_code'].append(codebook_indices)
            SOURCE_DATA_DICT['quant_shape'].append(quant_shape)
            SOURCE_DATA_DICT['slice_id'].append(batch['slice_id'].cpu().numpy())
            if task == 'Vid2Skel':
                SOURCE_DATA_DICT['image_sources'].append(batch['image_sources'])
            elif task == 'SkelPred':
                SOURCE_DATA_DICT['sources'].append(batch['sources'])


            pbar.update(1)
            num_iter += 1
            # if IS_DEBUG:
            #     if num_iter >= 32:
            #         break
            batch = prefetcher.next()
        pbar.close()

    for key in SOURCE_DATA_DICT:
        SOURCE_DATA_DICT[key] = np.concatenate(SOURCE_DATA_DICT[key], axis=0)
        print(f"{key}: shape={SOURCE_DATA_DICT[key].shape}")
    
    assert all([len(SOURCE_DATA_DICT[key])==len(SOURCE_DATA_DICT[list(SOURCE_DATA_DICT.keys())[0]]) for key in SOURCE_DATA_DICT])

    return SOURCE_DATA_DICT


if __name__ == "__main__":
    args = get_args()
    prompt_template_key = args.prompt_template
    task = args.task


    print('Loading dataset')
    # save_root_raw_data = osp.join(args.save_root, args.save_subdir_raw)
    save_root_raw_data = osp.join(args.save_root, args.save_subdir_raw, args.save_subdir_vqvae, args.save_subdir_jsonl)
    os.makedirs(save_root_raw_data, exist_ok=True)
    dataset_args_file = osp.join(save_root_raw_data, f'{args.data_split}_dataset_args.json')
    slice_indices_file = osp.join(save_root_raw_data, f'{args.data_split}_slice_indices.jsonl')
    if not osp.exists(dataset_args_file) or not osp.exists(slice_indices_file):
        dataset = load_dataset(args, dataset_args_file, slice_indices_file)
    else:
        with open(dataset_args_file, 'r') as f:
            dataset_args = json.load(f)
        dataset = Multimodal_Mocap_Dataset(
            max_samples=512 if IS_DEBUG else 999999999,
            **dataset_args
        )


    print('Getting code using vqvae')
    # save_root_vqvae = osp.join(args.save_root, args.save_subdir_raw, args.save_subdir_vqvae)
    save_root_vqvae = osp.join(args.save_root, args.save_subdir_raw, args.save_subdir_vqvae, args.save_subdir_jsonl)
    os.makedirs(save_root_vqvae, exist_ok=True)
    vqvae_output_file = osp.join(save_root_vqvae, f'{args.data_split}_vqvae_output.pkl')
    if not osp.exists(vqvae_output_file):
        SOURCE_DATA_DICT = get_skeleton_code(args, dataset, task)
        if not IS_DEBUG:
            joblib.dump(SOURCE_DATA_DICT, vqvae_output_file)
    else:
        SOURCE_DATA_DICT = joblib.load(vqvae_output_file)

    # if not IS_DEBUG:
    #     assert len(dataset) == len(SOURCE_DATA_DICT[list(SOURCE_DATA_DICT.keys())[0]])



    save_root_jsonl = osp.join(args.save_root, args.save_subdir_raw, args.save_subdir_vqvae, args.save_subdir_jsonl)
    os.makedirs(save_root_jsonl, exist_ok=True)

    jsonl_prompt_config_file = osp.join(save_root_jsonl, f'{args.data_split}_prompt_config.json')
    jsonl_prompt_config = {
        "task": task,
        "prompt_type": prompt_template_key,
        "get_skel_str_func": {
            "name": "get_skeleton_token_str_wTextualBodyPart_SplitByFrame",
            "input": "skeleton_indices",
        }
    }
    if not osp.exists(jsonl_prompt_config_file) and not IS_DEBUG:
        with open(jsonl_prompt_config_file, 'w') as f:
            json.dump(jsonl_prompt_config, f, indent=4)


    jsonl_file = osp.join(save_root_jsonl, f'{args.data_split}_data.jsonl')
    JSON_DATA_LIST = []
    for sample_id in tqdm(range(len(SOURCE_DATA_DICT[list(SOURCE_DATA_DICT.keys())[0]]))):
        data_item = edict.EasyDict(TASK_TEMPLATE[task])
        data_item.conversations[0]["value"] = f"{PROMPT_PLACEHOLDER}"
        
        # data_item.conversations[0]["value"] = f"<|task:{task}|>" + \
        #                                       f"<|prompt_type:{prompt_template_key}|>" + \
        #                                       f"{PROMPT_PLACEHOLDER}"
        # 记得修改 src/llamafactory/data/mm_plugin.py 中的 process_messages 函数, 添加 skeletons 的处理逻辑

        # TODO: 当 prompt 模板有多个备选, 在这里先提前选好, 不要把随机性放到 mmplugin 中
        # prompt_chosen_id = random.choice(PROMPT_TEMPLATES[task][prompt_template_key])
        # data_item.conversations[0]["value"] = f"<|task:{task}|>" + \
        #                                       f"<|prompt_type:{prompt_template_key}|>" + \
        #                                       f"<|prompt_chosen_id:{prompt_chosen_id}|>" + \
        #                                       f"{PROMPT_PLACEHOLDER}"

        slice_indices = SOURCE_DATA_DICT['slice_id'][sample_id]


        if task == 'Vid2Skel':
            image_sources_list = SOURCE_DATA_DICT['image_sources'][sample_id].tolist()
            image_sources_list = [image_src for image_src in image_sources_list]
            data_item.videos = [image_sources_list]
            skeleton_item = {
                'st_id': int(slice_indices[0]), 
                'ed_id': int(slice_indices[0-1]) + 1,
                'sample_id': sample_id,
                'data_key': args.joint_data_type,
                'data_aux_key': [args.joint_data_type.replace('normed', 'scale'), args.joint_data_type.replace('normed', 'transl'),
                                    'affine_trans', 'affine_trans_inv',
                                    'joint2d', 'joint2d_cpn',
                                    'factor_2_5d',
                                    ],
                'dataset_file': jsonl_file,  # 记得修改 src/llamafactory/data/mm_plugin.py 中的 process_messages 函数, 与这里保持一致
                # 'vqvae_output_file': vqvae_output_file,
                # 'dataset_args_file': dataset_args_file,
                # 'data_split': args.data_split,
            }
            data_item.skeletons = [skeleton_item]
        elif task == 'SkelPred':
            
            if sample_id == len(SOURCE_DATA_DICT['slice_id']) - 1:
                # the last sample, no next skeleton
                break
            sources = SOURCE_DATA_DICT['sources'][sample_id] # (T,)
            assert (sources[1:] == sources[:1]).all()
            source = sources[0]    # str
            next_sources = SOURCE_DATA_DICT['sources'][sample_id + 1] # (T,)
            assert (next_sources[1:] == next_sources[:1]).all()
            next_source = next_sources[0]    # str

            if source != next_source:
                # the next sample is from a different video
                continue

            next_slice_indices = SOURCE_DATA_DICT['slice_id'][sample_id + 1]

            skeleton_item = {
                'st_id': int(slice_indices[0]), 
                'ed_id': int(slice_indices[0-1]) + 1,
                'sample_id': sample_id,
                'data_key': args.joint_data_type,
                'data_aux_key': [args.joint_data_type.replace('normed', 'scale'), args.joint_data_type.replace('normed', 'transl'),
                                    'affine_trans', 'affine_trans_inv',
                                    'joint2d', 'joint2d_cpn',
                                    'factor_2_5d',
                                    ],
                'dataset_file': jsonl_file,  # 记得修改 src/llamafactory/data/mm_plugin.py 中的 process_messages 函数, 与这里保持一致
                # 'vqvae_output_file': vqvae_output_file,
                # 'dataset_args_file': dataset_args_file,
                # 'data_split': args.data_split,
            }
            next_skeleton_item = {
                'st_id': int(next_slice_indices[0]), 
                'ed_id': int(next_slice_indices[0-1]) + 1,
                'sample_id': sample_id + 1,
                'data_key': args.joint_data_type,
                'data_aux_key': [args.joint_data_type.replace('normed', 'scale'), args.joint_data_type.replace('normed', 'transl'),
                                    'affine_trans', 'affine_trans_inv',
                                    'joint2d', 'joint2d_cpn',
                                    'factor_2_5d',
                                    ],
                'dataset_file': jsonl_file,  # 记得修改 src/llamafactory/data/mm_plugin.py 中的 process_messages 函数, 与这里保持一致
                # 'vqvae_output_file': vqvae_output_file,
                # 'dataset_args_file': dataset_args_file,
                # 'data_split': args.data_split,
            }
            data_item.skeletons = [skeleton_item, next_skeleton_item]

        JSON_DATA_LIST.append(data_item)

    print(f"Saving to {jsonl_file}. Total {len(JSON_DATA_LIST)} samples.")
    with open(jsonl_file, 'w') as f:
        for item in JSON_DATA_LIST:
            f.write(json.dumps(item) + '\n')
