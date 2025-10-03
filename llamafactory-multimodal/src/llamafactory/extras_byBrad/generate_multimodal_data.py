import os
import os.path as osp
import torch
import numpy as np
import joblib
import easydict as edict
from tqdm import tqdm
import json 
import random
import argparse
from collections import defaultdict

from llamafactory.extras_byBrad.read_dataset import read_dataset
from llamafactory.extras_byBrad.vqvae_convert import convert_to_vqvae
from llamafactory.extras_byBrad.save_numpy_files import save_numpy_files
from llamafactory.extras_byBrad.prompt_templates import PROMPT_TEMPLATES
from llamafactory.extras_byBrad.utils import prepare_vqvae

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint_data_type', type=str, required=True, choices=['joint3d_image_normed', 'joint3d_image_affined_normed'])
    parser.add_argument('--prompt_template', type=str, required=True, choices=['fixed', 'simple', 'bodypart_aware', 'BodypartAwareExplicit', 'JointAwareExplicit'])
    parser.add_argument('--task', type=str, required=True, choices=['Vid2Skel', 'SkelPred', 'Text2Skel'])
    parser.add_argument('--data_split', type=str, required=True)

    parser.add_argument('--resume_pth', type=str, required=True)
    parser.add_argument('--save_root', type=str, default=osp.join(osp.abspath(__file__).split('LLaMA-Factory')[0], 'LLaMA-Factory/data/_multimodal_data_byBrad/'))
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--nb_code', type=int, default=8192)
    parser.add_argument('--codebook_dim', type=int, default=3072)

    parser.add_argument('--num_frames', type=int, default=None)
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

    parser.add_argument('--hrnet_output_level', type=int, default=None)
    parser.add_argument('--vision_guidance_ratio', type=float, default=None)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    vqvae = prepare_vqvae(args.resume_pth, args.joint_data_type)

    dataset = read_dataset(args)

    SOURCE_DATA_DICT = convert_to_vqvae(dataset, vqvae, args)

    save_numpy_files(SOURCE_DATA_DICT, args)

if __name__ == "__main__":
    main()