import re
import torch
import numpy as np
from typing import Optional, List

# 确保可以从 LLaMA-Factory 的根目录运行此脚本
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入您的 VQ-VAE 模型定义和相关组件
# 请确保这里的导入路径与您项目中的结构一致
from llamafactory.extras_byBrad.vqvae import Encoder, Decoder, VectorQuantizer, SKEL_VQVAE as SkeletonProcessor
from llamafactory.extras_byBrad.viz_skel_seq import viz_skel_seq_anim

# 导入 safetensors 加载工具
from safetensors.torch import load_file

# --- 定义与您训练时一致的特殊 Token ---
# 这些应该与您在 qwen2_5vl_lora_sft_byBrad.yaml 中定义的 add_special_tokens 一致
SKEL_START_TOKEN = "<|skel_start|>"
SKEL_END_TOKEN = "<|skel_end|>"
SKELETON_FRAME_BREAK = "<|frame_break|>"
# 这是您在 patcher.py 中动态添加的词元的基础格式
SKELETON_TOKEN_PATTERN = re.compile(r"<skel_(\d+)>")


def load_vqvae_model(checkpoint_path: str, codebook_size: int, device: str = "cpu") -> SkeletonProcessor:
    """
    从检查点加载 VQ-VAE 模型。
    
    注意：这里的模型参数 (如 mid_channels, code_dim 等) 
          必须与您训练 VQ-VAE 时使用的参数完全一致。
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"VQ-VAE checkpoint not found at: {checkpoint_path}")

    # --- VQ-VAE 初始化 (请确保参数匹配) ---
    encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1])
    vq = VectorQuantizer(nb_code=codebook_size, code_dim=3072, is_train=False)
    decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
    
    vqvae_model = SkeletonProcessor(encoder, decoder, vq)
    
    state_dict = load_file(checkpoint_path, device=device)
    vqvae_model.load_state_dict(state_dict)
    vqvae_model.to(device)
    vqvae_model.eval()
    
    return vqvae_model


def parse_skeleton_indices_from_text(text: str) -> Optional[List[List[int]]]:
    """
    从模型生成的文本中解析出骨架索引。
    返回一个二维列表，形如 [[frame1_joint_indices], [frame2_joint_indices], ...]。
    """
    # 1. 提取 <|skel_start|> 和 <|skel_end|> 之间的主要内容
    try:
        # 使用 re.DOTALL 使 '.' 可以匹配换行符
        main_block = re.search(f"{re.escape(SKEL_START_TOKEN)}(.*?){re.escape(SKEL_END_TOKEN)}", text, re.DOTALL).group(1)
    except AttributeError:
        # 如果没有找到匹配项
        return None

    # 2. 按帧分割
    frame_strings = main_block.split(SKELETON_FRAME_BREAK)

    # 3. 从每帧中解析出关节索引
    all_frames_indices = []
    for frame_str in frame_strings:
        if not frame_str:
            continue
        # 使用 re.findall 找到所有匹配 <skel_(\d+)> 的数字
        indices = [int(idx) for idx in SKELETON_TOKEN_PATTERN.findall(frame_str)]
        if indices:
            all_frames_indices.append(indices)

    return all_frames_indices if all_frames_indices else None


def decode_indices_to_skeleton(indices: List[List[int]], vqvae_decoder: SkeletonProcessor, device: str = "cpu") -> np.ndarray:
    """
    使用 VQ-VAE 解码器将解析出的索引转换为骨架 NumPy 数组。
    """
    # 1. 将 Python 列表转换为 PyTorch 张量
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    
    # 2. VQ-VAE 解码器通常需要一个批次维度，所以我们添加一个
    #    输入形状应为 [B, T_quant, J_quant], 这里 B=1
    indices_tensor = indices_tensor.unsqueeze(0)
    
    # 3. 调用解码器
    with torch.no_grad():
        # 假设 .decode() 方法接收索引并返回重建的骨架张量
        # 输出形状可能为 [B, C, T, J]
        reconstructed_skeleton_tensor = vqvae_decoder.decode(indices_tensor)

    # 4. 将张量转换回标准的 NumPy 格式
    #    从 [B, C, T, J] 转换为 [T, J, C]
    reconstructed_skeleton_tensor = reconstructed_skeleton_tensor.squeeze(0) # 移除批次维度 -> [C, T, J]
    reconstructed_skeleton_np = reconstructed_skeleton_tensor.cpu().numpy() # -> [T, J, C]
    
    return reconstructed_skeleton_np


if __name__ == "__main__":
    # --- 1. 配置 ---
    # !!! 重要: 请将此路径更新为您的 VQ-VAE 权重文件的真实路径 !!!
    VQVAE_CHECKPOINT_PATH = "/home/wxs/LLaMA-Factory/src/llamafactory/extras_byBrad/vqvae_experiment/all_datasets/models/checkpoint_epoch_113_step_500000/model.safetensors"
    CODEBOOK_SIZE = 8192
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2. 模拟的模型输出 ---
    # 这是一个包含了骨架序列的示例文本
    generated_text_from_model = (
        "当然！根据您的请求，我为您生成了一个挥手动作: "
        f"{SKEL_START_TOKEN}<skel_{1}><skel_{2}><skel_{3}><skel_{4}><skel_{5}><skel_{6}><skel_{7}><skel_{8}><skel_{9}><skel_{10}><skel_{11}><skel_{12}><skel_{13}><skel_{14}><skel_{15}><skel_{16}><skel_{17}>"
        f"{SKELETON_FRAME_BREAK}<skel_{1}><skel_{2}><skel_{3}><skel_{4}><skel_{5}><skel_{6}><skel_{7}><skel_{8}><skel_{9}><skel_{10}><skel_{11}><skel_{12}><skel_{13}><skel_{14}><skel_{15}><skel_{16}><skel_{17}>"
        f"{SKELETON_FRAME_BREAK}<skel_{1}><skel_{2}><skel_{3}><skel_{4}><skel_{5}><skel_{6}><skel_{7}><skel_{8}><skel_{9}><skel_{10}><skel_{11}><skel_{12}><skel_{13}><skel_{14}><skel_{15}><skel_{16}><skel_{17}>{SKEL_END_TOKEN}"
        " 希望对您有帮助！"
    )

    # --- 3. 执行解码流程 ---
    # 加载 VQ-VAE 模型
    print(f"Loading VQ-VAE model from {VQVAE_CHECKPOINT_PATH}...")
    vqvae_model = load_vqvae_model(VQVAE_CHECKPOINT_PATH, CODEBOOK_SIZE, device=DEVICE)
    print("VQ-VAE model loaded successfully.")

    # 从文本中解析索引
    print("\nParsing skeleton tokens from generated text...")
    parsed_indices = parse_skeleton_indices_from_text(generated_text_from_model)
    
    if parsed_indices:
        print(f"Successfully parsed indices for {len(parsed_indices)} frames.")
        
        # 将索引解码为骨架数据
        print("\nDecoding indices into skeleton data...")
        decoded_skeleton_data = decode_indices_to_skeleton(parsed_indices, vqvae_model, device=DEVICE)
        
        # 保存并验证输出
        output_path = "decoded_skeleton.npy"
        np.save(output_path, decoded_skeleton_data)
        print(f"\nSuccessfully decoded skeleton data with shape: {decoded_skeleton_data.shape}")
        print(f"Saved to '{output_path}'")
    else:
        print("Could not find a valid skeleton sequence in the generated text.")

    