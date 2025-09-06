import os
import torch
import numpy as np
import joblib
import easydict
from tqdm import tqdm
from PIL import Image
import json 
import random

from llamafactory.extras_byBrad.vqvae import SKEL_VQVAE as SkeletonProcessor, Encoder, VectorQuantizer, Decoder
from safetensors.torch import load_file

import sys
sys.path.append('/home/wxs/Skeleton-in-Context-tpami/')
from funcs_and_classes.Non_AR.dataset.ver13_ICL import DataReaderMesh
from lib.utils.utils_data import split_clips

num_frames=16
designated_split='test'

PROMPT_TEMPLATES = [
    "Describe the motion of the person in this video <video> using skeleton tokens. Focus on the motion of the whole body and the movement of body parts and joints over time.",
    "Generate the skeleton sequence corresponding to the person's movement in the video <video>.",
    "Transcribe the actions in the video <video> into a sequence of skeleton tokens.",
    "Output the skeleton tokens that represent the pose changes of the person shown in <video>.",
    "Please provide the skeletal representation for the movement in the video <video>.",
    "What is the skeleton token sequence for the person in <video>?",
    # 问题式
    "Can you show me the pose sequence for the person's movement in <video>?",
    "What would the motion capture data for the video <video> look like?",
    # 角色扮演式
    "You are a motion analysis expert. Analyze the video <video> and output the corresponding sequence of motion data.",
    # 填充式
    "Here is a video of a person moving: <video>. The corresponding structural representation is:",
    # 零样本/概念泛化
    "Analyze the person's movement in <video> and represent it structurally.",
]

TASK_TEMPLATE = {
    "conversations": [{"from": "human", "value": None},
                      {"from": "gpt", "value": "<skeleton>"}],
    "videos": [],
    "skeletons": []
    }

def img_to_skel():
    save_path = f'/home/wxs/LLaMA-Factory/data/source_data_byBrad/vid_to_skel/{designated_split}'
    jsonl_save_file = f'/home/wxs/LLaMA-Factory/data/custom_dataset_byBrad_vid2skel_{designated_split}.jsonl'

    load_data_file = "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl"
    load_image_source_file = "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/images_source.pkl"
    load_text_source_file = ""

    skeleton_processor = prepare_vqvae(mode='joint3d')
    img2skel_dataset = SkeletonDataset(num_frames=num_frames, data_mode='joint3d', designated_split=designated_split,
                                       load_data_file=load_data_file, load_image_source_file=load_image_source_file, load_text_source_file=load_text_source_file,
                                       return_extra=[['image']],
                                       )
    img2skel_dataloader = torch.utils.data.DataLoader(img2skel_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    POSES = []
    CODEBOOK_INDICES = []
    QUANT_SHAPES = []
    IMAGES = []
    for batch in tqdm(img2skel_dataloader):
        pose_seq, img_src = batch
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

        task_item = easydict.EasyDict(TASK_TEMPLATE)
        chosen_prompt = random.choice(PROMPT_TEMPLATES)
        task_item.conversations[0]["value"] = chosen_prompt
        task_item.videos = [IMAGES[sample_id]]
        task_item.skeletons = [codebook_index_save_file]

        jsonl_data.append(task_item)

    with open(jsonl_save_file, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')


def text_to_skel():
    save_path = f'/home/wxs/LLaMA-Factory/data/source_data_byBrad/text_to_skel/{designated_split}'
    jsonl_save_file = f'/home/wxs/LLaMA-Factory/data/custom_dataset_byBrad_text2skel_{designated_split}.jsonl'

    load_data_file = "/data2/wxs/DATASETS/AMASS_ByBradley/"
    load_image_source_file = ""
    load_text_source_file = "/data2/wxs/DATASETS/AMASS_ByBradley/retrieved_text_from_humanml3d.pkl"

    skeleton_processor = prepare_vqvae(mode='joint3d')
    text2skel_dataset = SkeletonDataset(num_frames=num_frames, data_mode='joint3d', designated_split=designated_split,
                                       load_data_file=load_data_file, load_image_source_file=load_image_source_file, load_text_source_file=load_text_source_file,
                                       return_extra=[['text']],
                                       )
    img2skel_dataloader = torch.utils.data.DataLoader(text2skel_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    POSES = []
    CODEBOOK_INDICES = []
    QUANT_SHAPES = []
    IMAGES = []
    for batch in tqdm(img2skel_dataloader):
        pose_seq, img_src = batch
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

        task_item = easydict.EasyDict(TASK_TEMPLATE)
        chosen_prompt = random.choice(PROMPT_TEMPLATES)
        task_item.conversations[0]["value"] = chosen_prompt
        task_item.videos = [IMAGES[sample_id]]
        task_item.skeletons = [codebook_index_save_file]

        jsonl_data.append(task_item)

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
    def __init__(self, num_frames=16, sample_stride=1, data_mode="joint3d", designated_split='train',
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
                                       'data_stride': num_frames,
                                       'read_confidence': False}
            datareader_config = {**datareader_config_unsplit, **datareader_config_split}
            datareader = DataReaderMesh(**datareader_config)        
            unsplit_data = DataReaderMesh.load_dataset_static(**datareader_config_unsplit)   # '/data2/wxs/DATASETS/AMASS_ByBradley'
            datareader.dt_dataset = unsplit_data

            read_func = datareader.read_2d if data_mode == "joint2d" else datareader.read_3d_image
            data_npy = read_func(designated_split=designated_split)     # (N,17,3)

            data_dict[dt_file] = {'poses': data_npy}

            if 'image' in extra_modality_list:
                img_list = joblib.load(img_src_file)[designated_split]
                img_list = img_list[::sample_stride]
                valid_img_indices = []
                for frame_id, img_path in enumerate(img_list):
                    if img_path is None:
                        continue
                    valid_img_indices.append(frame_id)
                    if use_cropped_image:
                        img_list[frame_id] = img_path.replace('images_fps50', f'images_fps50_cropped_{image_shape}')

                data_npy = data_npy[valid_img_indices]
                img_list = np.array(img_list)[valid_img_indices]
                assert len(img_list) == data_npy.shape[0]
            
                data_dict[dt_file]['img_src'] = img_list

                datareader.dt_dataset[designated_split]['source'] = np.array(datareader.dt_dataset[designated_split]['source'])[valid_img_indices].tolist()

            if 'text' in extra_modality_list:
                data_sources = datareader.read_source(designated_split=designated_split)
                text_data = joblib.load(txt_src_file)[designated_split]

                valid_text_indices = []
                for frame_id, source_str in enumerate(data_sources):
                    video_info, cam_info, frame_info = source_str.split('_')
                    video_id = int(video_info.replace('vid',''))
                    frame_indices_wrt_amass_npz = text_data[video_id]['frame_indices_wrt_amass_npz']
                    text_info_list = text_data[video_id]['humanml3d']
                    if (text_info_list) == 0:
                        continue
                    for text_info in text_info_list:
                        caption_path, valid_frame_indices_wrt_amass_npz = text_info







            split_id = datareader.get_split_id(designated_split=designated_split)   # 这里是用 unsplit_data 中的 'source' 来划分 split_id, 所以也要利用 valid_indices 作修改
            data_list.extend(zip([dt_file]*len(split_id), split_id))

        self.data_dict = data_dict
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        dt_file, slice_id = self.data_list[idx]
        poses = self.data_dict[dt_file]['poses'][slice_id]
        img_src = self.data_dict[dt_file]['img_src'][slice_id].tolist()
        return torch.from_numpy(poses).float(), img_src
    
def custom_collate_fn(batch):    
    poses_list = [item[0] for item in batch]
    img_src_list = [item[1] for item in batch]
    batched_poses = torch.stack(poses_list, dim=0)    
    return batched_poses, img_src_list

def prepare_vqvae(mode='joint3d'):
    encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1])
    vq = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)
    decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
    skeleton_processor = SkeletonProcessor(encoder, decoder, vq)

    if mode == 'joint3d':
        ckpt_path = "/home/wxs/LLaMA-Factory/src/llamafactory/extras_byBrad/vqvae_experiment/all_datasets/models/checkpoint_epoch_113_step_500000/model.safetensors"
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
    text_to_skel()
    # img_to_skel()