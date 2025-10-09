import os
import sys
from typing import Optional, Union, Tuple
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import math
from torch.nn.init import constant_, xavier_uniform_

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel, Qwen2_5_VLModelOutputWithPast, Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLTextModel,
    is_torchdynamo_compiling,
)
from transformers.utils import can_return_tuple, auto_docstring
from transformers.loss.loss_utils import fixed_cross_entropy, ForCausalLMLoss
from safetensors.torch import load_file
from llamafactory.extras_byBrad.vqvae import SKEL_VQVAE as SkeletonProcessor, Encoder, VectorQuantizer, Decoder
sys.path.append('../MTVCrafter/')
from models import HYBRID_VQVAE # type: ignore
sys.path.remove('../MTVCrafter/')

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.masking_utils import create_causal_mask
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import (GenerateNonBeamOutput, GenerationConfig, GenerateDecoderOnlyOutput)
from transformers.generation.streamers import BaseStreamer
from transformers.generation import GenerationMixin

logger = logging.get_logger(__name__)


class MultiScaleDeformableKeypointSampler(nn.Module):
    """
    一个更完整的、模仿Deformable DETR的可学习采样模块。
    - 多头 (n_heads): 学习不同的空间注意力模式。
    - 多点 (n_points): 在每个头内进行多尺度采样。
    - 带有精巧的参数初始化。
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        
        # 预测采样偏移量，每个点需要(x,y)两个坐标
        self.offset_predictor = nn.Linear(d_model, n_heads * n_points * 2)
        # 预测注意力权重，每个点一个权重
        self.weight_predictor = nn.Linear(d_model, n_heads * n_points)
        
        # 最终的输出投影层
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        对偏移量和权重预测器进行特定的初始化，以加速和稳定训练。
        """
        # 权重初始化为0，让模型从信任参考点开始
        constant_(self.offset_predictor.weight.data, 0.)
        
        # 偏置初始化为一个放射状的多尺度网格
        # 1. 在单位圆上均匀分布 n_heads 个方向
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        
        # 2. 将方向扩展为 n_points 个不同的尺度
        # grid_init shape: [n_heads, 2] -> [n_heads, 1, 2] -> [n_heads, n_points, 2]
        grid_init = grid_init.view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        
        # 3. 每个尺度上的点离中心越来越远
        for i in range(self.n_points):
            grid_init[:, i, :] *= i + 1
            
        # 4. 将这个精巧的初始偏移网格设置为偏置参数
        # 我们不希望初始偏移过大，所以乘以一个小的常数
        with torch.no_grad():
            self.offset_predictor.bias = nn.Parameter(0.01 * grid_init.view(-1))
            
        # 注意力权重预测器的权重和偏置初始化为0，使得初始权重均匀
        constant_(self.weight_predictor.weight.data, 0.)
        constant_(self.weight_predictor.bias.data, 0.)

        # 输出投影层使用标准的Xavier初始化
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, video_features, reference_points):
        """
        Args:
            video_features (Tensor): [B*T, C, H, W] 的视频特征图
            reference_points (Tensor): [B*T, J, 2] 的归一化参考点坐标 [-1, 1]
        
        Returns:
            Tensor: [B*T, J, C] 的加权采样特征
        """
        B_T, J, _ = reference_points.shape
        
        # 1. 生成查询向量(Query)
        initial_queries = F.grid_sample(
            video_features, reference_points.unsqueeze(2),
            align_corners=False, mode='bilinear'
        ).squeeze(-1).permute(0, 2, 1) # -> [B*T, J, C]
        
        # 2. 预测偏移量
        offsets = self.offset_predictor(initial_queries).view(B_T, J, self.n_heads, self.n_points, 2)
        
        # 3. 预测注意力权重
        # [B*T, J, C] -> [B*T, J, n_heads * n_points] -> softmax -> [B*T, J, n_heads, n_points, 1]
        weights = self.weight_predictor(initial_queries).view(B_T, J, self.n_heads * self.n_points)
        weights = torch.softmax(weights, -1).view(B_T, J, self.n_heads, self.n_points, 1)

        # 4. 计算最终的自适应采样坐标
        # reference_points [B*T, J, 2] -> [B*T, J, 1, 1, 2]
        # final_grid [B*T, J, n_heads, n_points, 2]
        final_grid = (reference_points.view(B_T, J, 1, 1, 2) + offsets).clamp(-1, 1)
        
        # 5. 在最终坐标上进行采样
        # final_grid [B*T, J, n_heads, n_points, 2] -> [B*T, J * n_heads * n_points, 1, 2]
        # video_features [B*T, C, H, W] -> [B*T, C, J*n_heads*n_points] after sampling
        # view -> [B*T, C, J, n_heads, n_points] -> permute -> [B*T, J, n_heads, n_points, C]
        sampled_features = F.grid_sample(
            video_features,
            final_grid.view(B_T, J * self.n_heads * self.n_points, 1, 2),
            align_corners=False,
            mode='bilinear'
        ).view(B_T, self.d_model, J, self.n_heads, self.n_points).permute(0, 2, 3, 4, 1)
        
        # 6. 对采样到的特征进行加权求和
        # ([B*T,J,n_heads,n_points,C] * [B*T,J,n_heads,n_points,1]).sum -> [B*T, J, C]
        output_features = (sampled_features * weights).sum(dim=(2, 3))
        
        # 7. 通过输出投影层并返回
        return self.output_proj(output_features)


class Qwen2_5_VLForConditionalGenerationWithSkeleton(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config: Qwen2_5_VLConfig, **kwargs_byBrad):
        super().__init__(config)

        ########## SKELETON ATTENTION PART #####################################################################################################################
        self.skeleton_attention_type = kwargs_byBrad['skeleton_attention_type']
        if self.skeleton_attention_type is not None:
            assert self.skeleton_attention_type in ['base', 'base_v2', 'nar', 'deformable_attn_w_joint2dcpn', 'deformable_attn_w_joint2dcpn_base']
        setattr(self.model, 'skeleton_attention_type', self.skeleton_attention_type)
        setattr(self.model.language_model, 'skeleton_attention_type', self.skeleton_attention_type)

        if self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn_base':
            self.model.deformable_sampler = MultiScaleDeformableKeypointSampler(
                d_model=config.hidden_size, 
                n_heads=8,      # 例如，8个头
                n_points=4,
            ) # n_heads可调


        ########## MPJPE EXRTA LOSS PART ##################################################################################################################### 
        self.use_mpjpe_loss = kwargs_byBrad['use_mpjpe_loss']
        self.mpjpe_success_count = 0

        ########## VQVAE PART ##################################################################################################################### 
        if 'vqvae_ckpt' in kwargs_byBrad:
            print('\n'.join(['Warning!!! `vqvae_ckpt` is deprecated, please use `vqvae_config` instead.' for _ in range(99)]))
            self.vqvae_ckpt = kwargs_byBrad['vqvae_ckpt']
        
            encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1])
            vq = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)
            decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
            self._skeleton_processor_container = [SkeletonProcessor(encoder, decoder, vq)]
        elif 'vqvae_config' in kwargs_byBrad:
            ################################################# ADDED BY BRADLEY 250917 #################################################
            vqvae_config = kwargs_byBrad['vqvae_config']
            self.vqvae_config = vqvae_config
            # vqvae_class = vqvae_config.vqvae_config.vqvae_class

            self.vqvae_ckpt = vqvae_config.vqvae_config.resume_path
            self._skeleton_processor_container = [HYBRID_VQVAE(vqvae_config.vqvae_config.encoder,
                                                               vqvae_config.vqvae_config.decoder, 
                                                               vqvae_config.vqvae_config.vq, 
                                                               vision_config=vqvae_config.vision_config, 
                                                               joint_data_type=vqvae_config.vqvae_config.joint_data_type)]

        if hasattr(self, '_skeleton_processor_container'):
            for param in self._skeleton_processor_container[0].parameters():
                param.requires_grad = False
            self._skeleton_processor_container[0].eval()

            self.is_vqvae_weights_loaded = False

        ########## FORWARD PART#####################################################################################################################
        self.model.forward = types.MethodType(custom_qwen2_5_vl_model_forward, self.model)
        self.model.language_model.forward = types.MethodType(custom_qwen2_5_vltextmodel_forward, self.model.language_model)

    def load_vqvae_weights(self):
        """
        加载 VQ-VAE 的权重到正确的设备上。
        """
        # 获取 skeleton_processor 当前所在的设备
        device = next(self.skeleton_processor.parameters()).device
        print(f"Loading VQ-VAE weights from {self.vqvae_ckpt} to device: {device}")
        
        # 从文件加载权重，直接加载到目标设备
        try:
            safetensors_path = os.path.join(self.vqvae_ckpt, "model.safetensors")
            pytorch_bin_path = os.path.join(self.vqvae_ckpt, "pytorch_model.bin")
            if os.path.exists(safetensors_path):
                print(f"Loading model from {safetensors_path}")
                state_dict = load_file(safetensors_path, device="cpu")
            elif os.path.exists(pytorch_bin_path):
                print(f"Loading model from {pytorch_bin_path}")
                state_dict = torch.load(pytorch_bin_path, map_location="cpu")
        except:
            state_dict = load_file(self.vqvae_ckpt, device=str(device))
        self.skeleton_processor.load_state_dict(state_dict, assign=True)
        self.is_vqvae_weights_loaded = True

    def to(self, device, *args, **kwargs):
        """
        重写 to 方法，以确保 skeleton_processor 也被移动到正确的设备。
        """
        # 首先，调用父类的 to 方法，移动模型的所有已注册参数
        super().to(device, *args, **kwargs)
        
        # 然后，手动将我们“隐藏”的 skeleton_processor 移动到相同的设备
        if hasattr(self, '_skeleton_processor_container'):
            processor = self._skeleton_processor_container[0]          
            # 检查 skeleton_processor 是否在 meta 设备上
            is_meta = any(p.is_meta for p in processor.parameters())
            
            if is_meta and device != torch.device("meta"):
                # 如果在 meta 设备上，不能直接 .to()。
                # 我们需要重新在目标设备上创建它。
                # 这会创建一个与原始模块结构相同但参数在目标设备上的新模块。
                print(f"Re-initializing SkeletonProcessor from meta to device: {device}")
                if not isinstance(processor, HYBRID_VQVAE):
                    print('\n'.join(['Warning!!! `SKEL_VQVAE` is deprecated, please use `HYBRID_VQVAE` instead.' for _ in range(99)]))
                    new_processor = type(processor)(
                        encoder=type(processor.encoder)(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1]),
                        decoder=type(processor.decoder)(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0]),
                        vq=type(processor.vq)(nb_code=8192, code_dim=3072, is_train=False),
                    ).to(device)
                else:
                    new_processor = type(processor)(
                        self.vqvae_config.vqvae_config.encoder,
                        self.vqvae_config.vqvae_config.decoder,
                        self.vqvae_config.vqvae_config.vq,
                        vision_config=self.vqvae_config.vision_config,
                        joint_data_type=self.vqvae_config.vqvae_config.joint_data_type
                    ).to(device)
                
                # 冻结参数并设置为评估模式
                for param in new_processor.parameters():
                    param.requires_grad = False
                new_processor.eval()
                
                # 替换掉原来的 meta 设备模块
                self._skeleton_processor_container[0] = new_processor
            else:
                # 如果不在 meta 设备上，或者目标设备也是 meta，正常移动
                processor.to(device)

            print(f"SkeletonProcessor is now on device: {next(self.skeleton_processor.parameters()).device}")

        # 确保返回 self 以支持链式调用
        return self
    
    @property
    def skeleton_processor(self):
        return self._skeleton_processor_container[0]

    def get_skeleton_placeholder_mask(self, input_ids, inputs_embeds):  # TODO. how to use this?
        """
        获取骨架数据在序列中占位符的掩码。
        """
        skeleton_token_id = getattr(self.config, "skeleton_token_id", -1)
        if skeleton_token_id == -1:
            raise ValueError("skeleton_token_id is not set in the model config.")
        
        special_skeleton_mask = (input_ids == skeleton_token_id)
        return special_skeleton_mask.unsqueeze(-1).expand_as(inputs_embeds)

    def _init_skeleton_parser(self):
        """
        初始化骨架解析器所需的映射。
        这个函数只在第一次需要时执行一次，以提高效率。
        """
        if hasattr(self, "_token_id_to_vq_index"):
            return

        skeleton_config = getattr(self.config, "skeleton_config", None)
        if skeleton_config is None:
            raise ValueError("`skeleton_config` not found in model config. Please set it during model loading.")

        # 创建一个从 token ID 到 VQ-VAE codebook 索引的映射
        skeleton_token_indices = skeleton_config['skeleton_token_indices']
        self._token_id_to_vq_index = {token_id: i for i, token_id in enumerate(skeleton_token_indices)}
        
        # 将所有骨架 token ID 存储在一个集合中，以便快速查找
        self._skeleton_token_id_set = set(skeleton_token_indices)

        self._skeleton_token_id_tensor = torch.tensor(
            skeleton_token_indices, device=self.device, dtype=torch.long
        )

    def _parse_skeleton_indices_from_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        从 token ID 序列中解析出骨架索引。
        使用 self.config.skeleton_config 中的信息。
        """
        self._init_skeleton_parser() # 确保映射已初始化

        # 遍历 token_ids，只收集属于骨架 token 的 ID
        # 注意：在 GPU 上直接进行这种循环效率较低，但对于辅助损失计算是可接受的
        vq_indices = [
            self._token_id_to_vq_index[token_id.item()]
            for token_id in token_ids
            if token_id.item() in self._skeleton_token_id_set
        ]

        if not vq_indices:
            return torch.tensor([], device=token_ids.device, dtype=torch.long)

        # 假设每 17 个索引为一帧。如果序列长度不完整，则丢弃末尾不完整的帧。
        num_frames = len(vq_indices) // 17
        if num_frames == 0:
            return torch.tensor([], device=token_ids.device, dtype=torch.long)
        
        # 截断并重塑为 [1, T, 17]
        vq_indices_tensor = torch.tensor(vq_indices[:num_frames * 17], device=token_ids.device, dtype=torch.long)
        return vq_indices_tensor.view(1, num_frames, 17)

    def compute_mpjpe_loss(self, pred_poses: torch.Tensor, true_poses: torch.Tensor) -> torch.Tensor:
        """
        计算 MPJPE (Mean Per Joint Position Error) 损失。
        Args:
            pred_poses (torch.Tensor): 预测的 3D 姿态，形状 (B, T, J, 3)。
            true_poses (torch.Tensor): 真实的 3D 姿态，形状 (B, T, J, 3)。
        Returns:
            torch.Tensor: 一个标量 MPJPE 损失值。
        """
        joint_errors = torch.norm(pred_poses - true_poses, p=2, dim=-1)
        mpjpe = torch.mean(joint_errors)
        return mpjpe
    
    def loss_function(  # copied from transformers.loss.loss_utils.ForCausalLMLoss
            self,
            logits,
            labels,
            vocab_size: int,
            num_items_in_batch: Optional[torch.Tensor] = None,
            ignore_index: int = -100,
            shift_labels: Optional[torch.Tensor] = None,
            **kwargs,
            ) -> torch.Tensor:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()

        if shift_labels is None:
            # Shift so that tokens < n predict n
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(logits.device)
        loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
        return loss
    
    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]):
        # from transformers.generation.utils import GenerationMixin
        try:
            return GenerationMixin._validate_model_kwargs(self, model_kwargs)
        except ValueError as valueerror:
            # print(valueerror)
            return


    # 重写 forward 方法以接收新的参数
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # --- 添加的新参数, 与 mmplugins 中的 regularize_skeletons 返回的字典的键保持一致 ----
        # skeleton_indices: Optional[torch.LongTensor] = None,  # 骨架数据的索引
        # skeleton_poses: Optional[torch.FloatTensor] = None,  # 骨架数据的3D姿态
        # skeleton_grid_thw: Optional[torch.LongTensor] = None,  # 骨架数据的网格尺寸
        # source_slice_id = None,  # 用于解码的源切片ID
        # joint2d_cpn_affined_normed = None,
        # --------------------
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        

        if 'skeleton_data_keys' in kwargs:
            skeleton_data_keys = kwargs.pop('skeleton_data_keys')
            skeleton_data_keys = skeleton_data_keys.item().split(',')
            skeleton_data_dict = {key: kwargs.pop(key) for key in skeleton_data_keys}



        if self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn' or self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn_base':
            if 'joint2d_cpn_affined_normed' in skeleton_data_dict.keys():
                raise NotImplementedError
            elif 'joint2d_cpn_affined' in skeleton_data_dict.keys():
                raise NotImplementedError
            elif 'joint2d_cpn' in skeleton_data_dict.keys():
                # src/llamafactory/data/mm_plugin.py::Qwen2VLPlugin._get_mm_inputs
                affine_trans = skeleton_data_dict['affine_trans']   # [B,T,2,3]
                vqvae_data_key = skeleton_data_dict['vqvae_data_key'].item()
                norm_scale = skeleton_data_dict[vqvae_data_key.replace('_normed', '_scale')].unsqueeze(-2)      # [B,T,1,3]
                norm_offset = skeleton_data_dict[vqvae_data_key.replace('_normed', '_transl')].unsqueeze(-2)    # [B,T,1,3]

                joint2d_cpn = skeleton_data_dict['joint2d_cpn']
                joint2d_cpn_xy1 = torch.cat([joint2d_cpn, joint2d_cpn.new_ones(joint2d_cpn[..., :1].shape)], dim=-1)    # [T,17,3]
                joint2d_cpn_affined = torch.einsum('btij,btkj->btik', joint2d_cpn_xy1, affine_trans)
                joint2d_cpn_affined_normed = joint2d_cpn_affined / norm_scale[..., :2] - norm_offset[..., :2]

                """Check alignment between 2d and 3d
                joint3d_image_affined_normed = skeleton_data_dict['joint3d_image_affined_normed']
                joint3d_image_affined = (joint3d_image_affined_normed + norm_offset) * norm_scale
                """

                skeleton_data_dict['joint2d_cpn_affined_normed'] = joint2d_cpn_affined_normed


        # --- 动态替换逻辑的起点 ---
        if self.training and self.skeleton_attention_type == 'nar':
            # 1. 从config获取所有骨架内容token ID和query token ID
            skeleton_token_indices = self.config.skeleton_config['skeleton_token_indices']
            query_token_ids = self.config.skeleton_config['skeleton_query_token_indices']
            
            # 创建一个副本以进行修改
            input_ids_modified = input_ids.clone()
            
            # 2. 遍历batch中的每个样本
            for i in range(input_ids.shape[0]):
                sequence = input_ids[i]
                
                # 3. 找到当前样本中所有骨架内容token的位置
                # is_skeleton_content_token是一个布尔张量
                is_skeleton_content_token = torch.isin(sequence, torch.tensor(skeleton_token_indices, device=input_ids.device))
                content_indices = torch.where(is_skeleton_content_token)[0]
                
                # 4. 按顺序替换
                # 将找到的第一个内容token替换为第一个query, 第二个替换为第二个...
                for j, token_position in enumerate(content_indices):
                    if j < len(query_token_ids):
                        input_ids_modified[i, token_position] = query_token_ids[j]
                    else:
                        # 如果骨架token数量超过了query数量，发出警告或报错
                        logger.warning_once("More skeleton tokens than available queries. Truncating.")
                        break
            
            # 5. 使用修改后的input_ids进行后续计算
            input_ids = input_ids_modified





        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )




        if self.skeleton_attention_type == 'base':
            batch_size, num_token = input_ids.shape
            skeleton_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for b in range(batch_size):
                ids = input_ids[b]
                # 找到所有 start/end 的位置
                start_indices = (ids == self.config.skeleton_config['skeleton_start_token_id']).nonzero(as_tuple=True)[0]
                end_indices = (ids == self.config.skeleton_config['skeleton_end_token_id']).nonzero(as_tuple=True)[0]
                # 一一配对
                for start, end in zip(start_indices, end_indices):
                    skeleton_mask[b, start:end+1] = True            
            kwargs['skeleton_mask'] = skeleton_mask

        

        
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            skeleton_data_dict=skeleton_data_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:


            if self.skeleton_attention_type == 'nar':
                assert not self.use_mpjpe_loss
                # =================================================================================================
                # ================================ MODIFIED BY GEMINI: START ======================================
                # =================================================================================================
                #                      --- Hybrid Causal & Non-Causal Loss Calculation ---
                
                # 假设: 您的配置文件中包含了 query token 的 ID 列表
                # 例如: self.config.skeleton_config['query_token_ids'] = [id1, id2, ...]
                # 为确保代码健壮性，我们使用 getattr 安全地获取它
                query_token_config = self.config.skeleton_config['skeleton_query_token_indices']
                
                # 初始化两个损失项
                loss_causal = None
                loss_non_causal = None

                # --- 1. 计算 Non-Causal Loss (用于骨架内容填充) ---
                query_token_ids = torch.tensor(query_token_config, device=input_ids.device)
                
                # is_query_mask: [B, L], bool, 标记出 input_ids 中所有 query token 的位置
                is_query_mask = torch.isin(input_ids, query_token_ids)

                if torch.any(is_query_mask):
                    # 在 query 位置，labels 张量应该包含直接的、非移位的目标 (s_i)
                    # 我们只在这些位置计算非因果损失
                    
                    # 从 logits 中提取 query 位置的预测
                    query_logits = logits[is_query_mask]
                    
                    # 从 labels 中提取 query 位置的真实目标
                    query_labels = labels[is_query_mask]
                    '<skel_316><skel_649>...<skel_2752><skel_1835>'
                    # 直接计算交叉熵，没有任何 "shift"
                    loss_non_causal = nn.functional.cross_entropy(
                        query_logits.float(), 
                        query_labels, 
                        ignore_index=-100
                    )

                # --- 2. 计算 Causal Loss (用于结构生成) ---
                # 为了计算因果损失，我们要屏蔽掉非因果的部分，避免它们干扰
                causal_labels = labels.clone()
                if 'is_query_mask' in locals() and torch.any(is_query_mask):
                    causal_labels[is_query_mask] = -100  # 在计算因果损失时，忽略所有 query 的位置

                # 使用您原来的、带内部移位的标准损失函数
                loss_causal = self.loss_function(logits=logits, labels=causal_labels, vocab_size=self.config.vocab_size)
                #     INPUT                LABEL
                # \n                    <|skel_start|>
                # <skel_query_67>       <|skel_end|>
                # <|skel_end|>          <|im_end|>
                # <|im_end|>            \n

                # --- 3. 合并损失 ---
                # 我们从因果损失开始
                total_loss = loss_causal if loss_causal is not None and torch.isfinite(loss_causal) else 0.0
                
                # 如果存在非因果损失，则按权重加上它
                if loss_non_causal is not None and torch.isfinite(loss_non_causal):
                    # 建议在 config 中定义一个权重，用于平衡两种损失
                    non_causal_weight = getattr(self.config, "non_causal_loss_weight", 1.0)
                    total_loss = total_loss + non_causal_weight * loss_non_causal

                # 将最终计算出的 total_loss 赋给 loss
                loss = total_loss if total_loss != 0.0 else None
                # =================================================================================================
                # ================================= MODIFIED BY GEMINI: END =======================================
                # =================================================================================================



            else:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)  # ForCausalLMLoss
                # logits: [1,352,159872]; labels: [1,352]

                # ADDED BY BRADLEY 250911 #################################################################################
                mpjpe_loss = None
                if self.use_mpjpe_loss and skeleton_poses is not None:
                    try:
                        if not self.is_vqvae_weights_loaded:
                            self.load_vqvae_weights()

                        self._init_skeleton_parser() # 确保解析器已初始化

                        # 步骤 1: 识别真实 `labels` 中的骨架 token 区域
                        # is_gt_skeleton_mask: [B, L], bool, 标记哪些位置应该是骨架 token
                        is_gt_skeleton_mask = torch.isin(labels, self._skeleton_token_id_tensor)   # [1,352]

                        # 如果当前批次中没有任何骨架 token，则直接跳过
                        if torch.any(is_gt_skeleton_mask):
                            # 步骤 2: 获取模型在这些真实骨架区域的预测
                            # pred_ids_in_gt_skel_region: [N], N 是骨架 token 的总数
                            skel_logits = logits[is_gt_skeleton_mask]

                            # 步骤 3: (可微分方式) 计算期望的VQ-VAE嵌入
                            # 3.1 仅提取与骨架token相关的logits
                            # skeleton_token_logits: [N, num_skeleton_tokens]
                            skeleton_token_logits = skel_logits[:, self._skeleton_token_id_tensor]

                            # 3.2 计算这些骨架token的概率分布
                            # skeleton_probs: [N, num_skeleton_tokens]
                            skeleton_probs = torch.softmax(skeleton_token_logits, dim=-1)

                            # 3.3 获取VQ-VAE码本(codebook)的嵌入向量
                            # vq_embeddings: [num_skeleton_tokens, code_dim]
                            # 注意: VQ-VAE的参数需要是可访问的，并且不参与梯度更新
                            vq_embeddings = self.skeleton_processor.vq.codebook.detach()

                            # 3.4 通过加权平均计算期望的嵌入
                            # expected_embedding: [N, code_dim]
                            expected_embedding = torch.matmul(skeleton_probs, vq_embeddings)

                            # 步骤 4: 解码期望嵌入以获得3D姿态
                            # 假设每17个token为一帧
                            num_frames = expected_embedding.size(0) // 17
                            if num_frames > 0:
                                # 截断并重塑以匹配解码器输入
                                num_tokens_to_decode = num_frames * 17
                                quantized = expected_embedding[:num_tokens_to_decode].view(1, num_frames, 17, -1)
                                
                                # 使用解码器直接从量化嵌入解码姿态
                                # 注意: skeleton_processor.decode需要能处理嵌入向量
                                # 如果它只能处理索引，则需要修改或使用其内部的解码器部分
                                # 这里我们假设可以直接调用解码器
                                pred_poses = self.skeleton_processor.decode_from_quantized(quantized)
                                pred_poses = pred_poses.permute(0, 2, 3, 1) # [B, T, J, 3]
                                # 裁剪真实姿态以匹配预测的长度
                                T_pred = pred_poses.size(1)
                                true_poses_sliced = skeleton_poses[0][None, :T_pred, :, :]

                                # 计算 MPJPE 损失
                                if true_poses_sliced.size(1) == T_pred:
                                    mpjpe_loss = self.compute_mpjpe_loss(pred_poses, true_poses_sliced)


                    except Exception as e:
                        print(f"Could not compute MPJPE loss: {e}")
                        mpjpe_loss = None
                ###########################################################################################################

                if mpjpe_loss is not None and torch.isfinite(mpjpe_loss):
                    mpjpe_loss_weight = getattr(self.config, "mpjpe_loss_weight", 0.1)
                    loss = loss + mpjpe_loss_weight * mpjpe_loss
                    self.mpjpe_success_count += 1

                    # 2. 添加条件打印逻辑
                    # 只在主进程 (rank 0) 打印，避免日志混乱
                    # 在第一次成功时打印，之后每100次成功打印一次
                    if int(os.getenv("LOCAL_RANK", "0")) == 0:
                        if self.mpjpe_success_count == 1 or self.mpjpe_success_count % 100 == 0:
                            print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>> [Rank 0] MPJPE loss successfully applied <<<<<<<<<<<<<<<<<<<<<<<<<<")
                            print(f">>>>>>>>>>>>>>>>>>>>>>>>>> Total application count: {self.mpjpe_success_count} <<<<<<<<<<<<<<<<<<<<<<<<<<\n")


        final_output = Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
        
        if hasattr(outputs, 'skeleton_visual_features'):
            final_output['skeleton_visual_features'] = outputs.skeleton_visual_features
        
        if hasattr(outputs, 'skeleton_token_index'):
            final_output['skeleton_token_index'] = outputs.skeleton_token_index


        return final_output
    

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )
        return model_inputs

        


    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        """
        重载此方法以注入混合AR-NAR的attention_mask更新逻辑。
        """
        # 1. 首先，调用父类的原始方法来处理标准更新（例如KV缓存）
        # 这样可以确保我们不会破坏任何基类功能

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens=num_new_tokens
        )

        if self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn':
            if hasattr(outputs, 'skeleton_visual_features'):
                model_kwargs['skeleton_visual_features'] = outputs.skeleton_visual_features
            
            if hasattr(outputs, 'skeleton_token_index'):
                # 注意：这里我们应该从原始的 kwargs 中获取更新后的 index
                # 因为 prepare_inputs_for_generation 更新的是 kwargs 字典
                model_kwargs['skeleton_token_index'] = model_kwargs.get('skeleton_token_index', 0)

        elif self.skeleton_attention_type != 'nar':
            return model_kwargs
        
        # next_is_pose_block = model_kwargs.get("next_is_pose_block", False)

        # # 2. 关键：根据是否进入姿态生成模式，来扩展 attention_mask
        # if "attention_mask" in model_kwargs:
        #     attention_mask = model_kwargs["attention_mask"]
        #     if next_is_pose_block:
        #         # 如果下一轮是姿态生成，拼接 num_pose_tokens 个值为 2 的信号

        #         # num_pose_tokens = 80 # 您可以根据VQVAE的设置调整.
        #         # TODO
        #         # num_pose_tokens 这个数字必须与您在config.json中定义的query_token_ids列表的长度严格相等。如果未来您调整了配置（比如改成40或100个查询token），但忘记修改这里的80，生成过程将会出错且难以调试。
        #         # 修复建议: 让配置文件成为唯一的真实来源 (Single Source of Truth) 
        #         num_pose_tokens = len(self.config.skeleton_config['skeleton_query_token_indices'])
                
        #         pose_mask_signal = torch.full(
        #             (attention_mask.shape[0], num_pose_tokens), 
        #             2, 
        #             device=attention_mask.device, 
        #             dtype=attention_mask.dtype
        #         )
        #         attention_mask = torch.cat([attention_mask, pose_mask_signal], dim=-1)
        #         model_kwargs['attention_mask'] = attention_mask
            
        return model_kwargs


    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        generation_config: Optional[GenerationConfig] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

        if self.skeleton_attention_type != 'nar':
            return GenerationMixin._sample(
                self, input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus, streamer, **model_kwargs
            )
        
        # --- 标准初始化 ---
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        do_sample = generation_config.do_sample
        return_dict_in_generate = generation_config.return_dict_in_generate
        scores = () if (return_dict_in_generate and generation_config.output_scores) else None
        raw_logits = () if (return_dict_in_generate and generation_config.output_logits) else None

        # --- AR-NAR 专用初始化 ---
        skel_start_id = self.config.skeleton_config['skeleton_start_token_id']
        skel_end_id = self.config.skeleton_config['skeleton_end_token_id']
        query_token_ids = self.config.skeleton_config.get('skeleton_query_token_indices')
        if query_token_ids is None:
            raise ValueError("`query_token_ids` must be defined in `config.skeleton_config` for NAR generation.")
        num_pose_tokens = len(query_token_ids)
        query_tokens_tensor = torch.tensor(query_token_ids, device=input_ids.device, dtype=torch.long)

        batch_size, cur_len = input_ids.shape[:2]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        
        # --- 核心生成循环 ---
        iter_cnt = 0
        while self._has_unfinished_sequences(False, synced_gpus, device=input_ids.device):
            # 检查是否正要进入NAR步骤 (这个标志由上一轮循环末尾设置)
            is_entering_nar_step = model_kwargs.pop("next_is_pose_block", False)

            # --- 【核心修复】在循环顶部准备好所有输入 ---
            if is_entering_nar_step:
                original_rope_deltas = self.model.rope_deltas
                # ================================ MODIFIED BY GEMINI: START ================================
                #                      --- 【最终BUG修复】为NAR步骤禁用KV缓存 ---
                # 这是强制模型执行完整前向传播并返回所有logits的关键
                model_kwargs["past_key_values"] = None
                # ================================ MODIFIED BY GEMINI: END ==================================
                # 扩展 input_ids
                current_batch_size = input_ids.shape[0]
                input_ids = torch.cat([input_ids, query_tokens_tensor.expand(current_batch_size, -1)], dim=-1)




                input_ids_suffix = torch.tensor([[155830, 151645,    198, 151643, 151643, 151643, 151643, 151643, 151643, 151643]]).to(input_ids.device).to(input_ids.dtype)
                # '<|skel_end|><|im_end|>\n<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>'
                input_ids = torch.cat([input_ids, input_ids_suffix], dim=-1)
                




                
                # 扩展 attention_mask
                attention_mask = model_kwargs.get("attention_mask")

                attention_mask[:, -1] = 2   # <|skel_start|>
                pose_mask_signal = torch.full(
                    (attention_mask.shape[0], num_pose_tokens), 2,
                    device=attention_mask.device, dtype=attention_mask.dtype
                )



                attention_mask_suffix = torch.tensor([[2, 1, 1, 0, 0, 0, 0, 0, 0, 0]]).to(attention_mask.device).to(attention_mask.dtype)






                attention_mask = torch.cat([attention_mask, pose_mask_signal, attention_mask_suffix], dim=-1)
                model_kwargs['attention_mask'] = (attention_mask != 0).int()


                # 3. 【关键】手动为这个新的“预填充”步骤创建正确的 cache_position
                # 它的长度必须与我们刚刚构建的完整 input_ids 的长度一致
                model_kwargs["cache_position"] = torch.arange(input_ids.shape[1], device=input_ids.device)

                # ================================ MODIFIED BY GEMINI: START ================================
                #                      --- 【关键修复】为NAR推理禁用Logits切片 ---
                # 明确告诉forward函数，在这次NAR前向传播中，不要切片，返回所有logits
                model_kwargs["logits_to_keep"] = slice(None)
                # ================================ MODIFIED BY GEMINI: END ==================================


            # 准备模型输入
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_entering_nar_step:
                model_inputs['attention_mask'] = attention_mask
            
            # 前向传播
            outputs = self(**model_inputs, return_dict=True)

            if is_entering_nar_step:
                self.model.rope_deltas = original_rope_deltas

            # 提取 logits
            if is_entering_nar_step:
                next_token_logits = outputs.logits[:, -num_pose_tokens:, :]
            else:
                next_token_logits = outputs.logits[:, -1, :]

            # Logits 处理和采样
            if next_token_logits.ndim == 3: # NAR
                processed_scores = []
                for i in range(next_token_logits.shape[1]):
                    # 上下文 input_ids 对于这个块中的所有token都是一样的
                    step_logits = next_token_logits[:, i, :]
                    step_scores = logits_processor(input_ids, step_logits)
                    processed_scores.append(step_scores.unsqueeze(1))
                next_token_scores = torch.cat(processed_scores, dim=1)
            else: # AR
                next_token_scores = logits_processor(input_ids, next_token_logits)

            if do_sample:
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                probs = probs.view(batch_size, -1, probs.shape[-1])
                next_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(batch_size, probs.shape[1])
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # 更新序列和下一轮的状态标志
            next_is_pose_block_for_next_iter = False
            if next_tokens.shape[1] == 1 and torch.all(next_tokens == skel_start_id):
                next_is_pose_block_for_next_iter = True
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            elif next_tokens.shape[1] == num_pose_tokens:
                input_ids[:, -num_pose_tokens:] = next_tokens
                end_token = torch.full((batch_size, 1), skel_end_id, device=input_ids.device, dtype=torch.long)
                input_ids = torch.cat([input_ids, end_token], dim=-1)
            else:
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # 更新KV缓存和下一轮的状态
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder, num_new_tokens=next_tokens.shape[1]
            )
            model_kwargs["next_is_pose_block"] = next_is_pose_block_for_next_iter
            
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            
            # 停止条件检查
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            if this_peer_finished and not synced_gpus:
                break

            iter_cnt += 1
        
        # ... (返回逻辑)
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=input_ids, past_key_values=model_kwargs.get("past_key_values"))
        else:
            return input_ids


# 确保函数签名与原始 `Qwen2_5_VLModel.forward` 方法完全相同
def custom_qwen2_5_vl_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    skeleton_data_dict = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        # transformers.models.qwen2_5_vl.modeling_qwen2_5_vl::Qwen2_5_VLModel.get_video_features
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)



        # MODIFIED BY GEMINI: 移除了在这里创建和传递 video_mask_unpad 的逻辑。
        # 它将在 custom_qwen2_5_vltextmodel_forward 中被统一创建和处理。
        if self.skeleton_attention_type == 'base':
            assert (video_mask[:,:,:1] == video_mask[:,:,1:]).all()
            video_mask_unpad = video_mask[:,:,0]    # [1,352]
            kwargs['video_mask_unpad'] = video_mask_unpad
        

        if self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn' or self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn_base':
            joint2d_coords = skeleton_data_dict['joint2d_cpn_affined_normed']
            
            video_embeds_shape_unflatten = video_grid_thw // torch.tensor([1, self.visual.spatial_merge_size, self.visual.spatial_merge_size]).to(video_grid_thw.device, video_grid_thw.dtype)
            
            # video_embeds.split(video_embeds_shape_unflatten.prod(-1).tolist(), dim=0)
            B = video_embeds_shape_unflatten.shape[0]
            T_merged, H, W = video_embeds_shape_unflatten[0].tolist()
            C = video_embeds.shape[-1]
            T, J = joint2d_coords.shape[1:3]
            
            video_embeds_unflatten = video_embeds.reshape(B, T_merged, H, W, C)  # [B, T, H', W', C]

            # ================================ MODIFIED BY GEMINI: START ================================
            # --- 1.5 Align Temporal Dimensions ---
            # The video encoder merges frames (T -> T/2), so we must align the coordinates.
            # We downsample the coordinates by averaging every two consecutive frames.
            
            aligned_joint2d_coords = None
            if T == 2 * T_merged:
                # Reshape to group frames by pairs: [B, T, J, 2] -> [B, T_merged, 2, J, 2]
                coords_paired = joint2d_coords.view(B, T_merged, 2, J, -1)
                # Average the pairs to get aligned coords: [B, T_merged, J, 2]
                aligned_joint2d_coords = torch.mean(coords_paired, dim=2)
            # elif T == T_merged:
            #     # If by chance they are already aligned, just use them directly.
            #     aligned_joint2d_coords = joint2d_coords
            else:
                raise ValueError(f"Temporal dimension mismatch: video T_merged={T_merged}, joint2d T={T}. Expected joint2d T to be double the video T_merged")

            if self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn_base':
                video_features_for_sampling = video_embeds_unflatten.reshape(B * T_merged, H, W, C).permute(0, 3, 1, 2)
                # reference_points 需要是 [-1, 1] 范围
                reference_points_for_sampler = aligned_joint2d_coords.view(B * T_merged, J, 2)

                # === 核心修改：调用可学习的采样器 ===
                # 用一行代码取代了之前固定的 grid_sample 逻辑
                skeleton_visual_features = self.deformable_sampler(
                    video_features_for_sampling,
                    reference_points_for_sampler
                ).view(B, T_merged * J, C)


                global_pose_context = torch.mean(skeleton_visual_features, dim=1, keepdim=True)
                
                # --- C. 将全局上下文向量叠加到原始视频特征上 ---
                # video_embeds 的形状是 [num_video_tokens, C]，其中 num_video_tokens = T_merged * H * W
                # global_pose_context 的形状是 [1, 1, C] (假设B=1)
                # 通过PyTorch的广播机制，这个全局向量会被加到每一个原始视频特征上
                video_embeds = video_embeds.view(B, -1, C)
                video_embeds = video_embeds + global_pose_context
                # 恢复 video_embeds 的原始形状
                video_embeds = video_embeds.view(-1, C)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                logger.info_once("Successfully enhanced video features with global pose context.")



            elif self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn':
                is_prefill_or_training = (past_key_values is None or past_key_values.get_seq_length() == 0)
                if is_prefill_or_training:
                    # --- 2. Prepare Inputs for Grid Sampling ---
                    # Permute video features to match grid_sample's expected input: [B*T, C, H, W]
                    video_features_for_sampling = video_embeds_unflatten.reshape(B * T_merged, H, W, C).permute(0, 3, 1, 2)
                    
                    # ================================  CORRECTED LINE  ================================
                    # Reshape 2D coordinates directly as they are already in the [-1, 1] range.
                    sampling_grid = aligned_joint2d_coords.reshape(B * T_merged, J, 2)
                    # =================================================================================
                    
                    # Add a dummy dimension for grid_sample's 'grid' argument: [B*T, J, 1, 2]
                    sampling_grid = sampling_grid.unsqueeze(2)

                    # --- 3. Perform Guided Sampling ---
                    sampled_features = F.grid_sample(
                        video_features_for_sampling,
                        sampling_grid,
                        align_corners=False, # Should be False for feature maps
                        mode='bilinear',
                        padding_mode='zeros'
                    )  # Output shape: [B*T, C, J, 1]

                    # ... (The rest of the logic for reshaping and injecting features remains the same) ...
                    
                    # --- 4. Reshape Sampled Features for Injection ---
                    sampled_features = sampled_features.squeeze(-1).permute(0, 2, 1)
                    skeleton_visual_features = sampled_features.reshape(B, T_merged * J, C)


                    # 核心：将计算结果放入kwargs，以便在generate循环中传递
                    kwargs['skeleton_visual_features'] = skeleton_visual_features
                    kwargs['skeleton_token_index'] = 0



                    if self.training:
                        # --- 5. Identify Skeleton Placeholder Tokens and Inject Features ---
                        skel_token_ids = self.config.skeleton_config['skeleton_token_indices']
                        skel_start_id = self.config.skeleton_config['skeleton_start_token_id']
                        skel_end_id = self.config.skeleton_config['skeleton_end_token_id']

                        is_skel_token = torch.isin(input_ids, torch.tensor(skel_token_ids, device=input_ids.device))
                        is_not_boundary = (input_ids != skel_start_id) & (input_ids != skel_end_id)
                        skeleton_content_mask = is_skel_token & is_not_boundary

                        inputs_embeds = inputs_embeds.masked_scatter(
                            skeleton_content_mask.unsqueeze(-1), skeleton_visual_features
                        )
                        
                        if not hasattr(self, '_deformable_attn_applied_once'):
                            logger.info("Successfully applied keypoint-guided feature sampling.")
                            self._deformable_attn_applied_once = True






    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.language_model(
        input_ids=input_ids, # MODIFIED BY GEMINI: 将 input_ids 传递下去，用于mask创建
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        skeleton_data_dict=skeleton_data_dict,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


    if self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn':
        if 'skeleton_visual_features' in kwargs:
            # 3. 【核心】将数据附加到 outputs 对象上
            # ModelOutput 对象支持动态设置属性
            setattr(output, 'skeleton_visual_features', kwargs['skeleton_visual_features'])
            setattr(output, 'skeleton_token_index', kwargs['skeleton_token_index'])


    return output if return_dict else output.to_tuple()




def custom_qwen2_5_vltextmodel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    skeleton_data_dict = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # MODIFIED BY GEMINI: 修改了这里的逻辑，因为我们需要 input_ids 来创建 mask
    if input_ids is None and inputs_embeds is None:
        raise ValueError("You must specify at least one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    
    # MODIFIED BY GEMINI: 从 embeds 推断 input_ids, 以防外部只提供了 embeds
    # 这是创建 mask 的一个后备方案，但可能不准确。最佳实践是始终提供 input_ids。
    if input_ids is None:
        raise NotImplementedError
        logger.warning_once("`input_ids` is not provided. Custom attention mask may not be correctly applied.")
        # 如果没有 input_ids, 我们无法创建自定义mask，只能回退到默认行为
        input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)


    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        # if self.has_sliding_layers:
        #     causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None



    # ================================ MODIFIED BY GEMINI: START ================================
    #                      --- 【最终BUG修复】处理 SDPA 优化导致的 None Mask ---
    
    base_causal_mask = causal_mask_mapping.get("full_attention")

    # 关键检查: 如果基础遮罩是 None (因为SDPA优化)，但我们又需要一个自定义遮罩，
    # 那么我们必须自己构建一个标准的4D因果遮罩作为修改的起点。
    if base_causal_mask is None and self.skeleton_attention_type is not None:
        # 这个导入是必要的，它包含了构建4D遮罩的标准工具
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

        logger.warning_once(
            "Base causal mask is None, likely due to SDPA optimization. "
            "Manually creating a 4D causal mask for custom attention modification."
        )
        # 从 kwargs 获取 2D attention_mask，如果不存在则创建一个
        _2d_attention_mask = mask_kwargs.get("attention_mask")
        if _2d_attention_mask is None:
             _2d_attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
            )
        
        # 调用 transformers 的标准内部函数来构建4D遮罩
        base_causal_mask = _prepare_4d_causal_attention_mask(
            _2d_attention_mask,
            inputs_embeds.shape[:2],
            inputs_embeds,
            0, # past_key_values_length, 在 pre-fill 步骤中为 0
        )

        base_causal_mask = (base_causal_mask == 0)


    if self.skeleton_attention_type == 'base':
        skeleton_mask = kwargs.pop('skeleton_mask')
        skeleton_mask = skeleton_mask.int()
        skeleton_attention_mask = torch.einsum('bq,bk->bqk', skeleton_mask, skeleton_mask)
        skeleton_attention_mask = skeleton_attention_mask.unsqueeze(1).bool()

        video_mask_unpad = kwargs.pop('video_mask_unpad')
        skeleton_video_cross_attention_mask = torch.einsum('bq,bk->bqk', skeleton_mask, video_mask_unpad)
        skeleton_video_cross_attention_mask = skeleton_video_cross_attention_mask.unsqueeze(1).bool()

        causal_mask_mapping['full_attention'] = causal_mask_mapping['full_attention'] | skeleton_attention_mask | skeleton_video_cross_attention_mask


        if not hasattr(self, 'skeleton_attention_applied_flag'):
            self.skeleton_attention_applied_flag = True
            print('\n'.join([f'skeleton_attention <{self.skeleton_attention_type}> successfully applied' for _ in range(99)]))

    elif self.skeleton_attention_type == 'base_v2':
        # =================================================================================================
        # ================================ MODIFIED BY GEMINI: START ======================================
        # =================================================================================================
        # 使用 hasattr 检查，以确保在没有设置此属性的模型上不会出错
        # 调用我们创建的辅助函数来生成完整的、正确的遮罩
        final_mask = create_custom_pose_attention_mask_baseV2(
            input_ids=input_ids,
            base_causal_mask=causal_mask_mapping['full_attention'],
            config=self.config,       # self.config 是从主模型传递过来的完整配置
            is_training=self.training # self.training 可以正确判断当前是训练还是推理
        )
        causal_mask_mapping['full_attention'] = final_mask
        # =================================================================================================
        # ================================= MODIFIED BY GEMINI: END =======================================
        # =================================================================================================
        if not hasattr(self, 'skeleton_attention_applied_flag'):
            self.skeleton_attention_applied_flag = True
            print('\n'.join([f'skeleton_attention <{self.skeleton_attention_type}> successfully applied' for _ in range(99)]))



    elif self.skeleton_attention_type == 'nar':

        full_input_ids = kwargs.get("full_input_ids_for_mask", input_ids)

        # 它接收1D的"信号"mask (mask_kwargs['attention_mask'])
        # 和基础的4D因果mask (causal_mask_mapping['full_attention'])
        final_mask = create_custom_pose_attention_mask_nar(
            # input_ids=input_ids,
            input_ids=full_input_ids,
            attention_mask=mask_kwargs['attention_mask'], # <--- 读取信号
            base_causal_mask=base_causal_mask,
            config=self.config,
            is_training=self.training
        )
        
        # 3. 它输出一个构建好的、可供执行的4D mask
        causal_mask_mapping['full_attention'] = final_mask
        if not hasattr(self, 'skeleton_attention_applied_flag'):
            self.skeleton_attention_applied_flag = True
            print('\n'.join([f'skeleton_attention <{self.skeleton_attention_type}> successfully applied' for _ in range(99)]))




    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        # decoder_layer.self_attn 中的 attention_interface 的实现: transformers/integrations/sdpa_attention.py::sdpa_attention_forward
        # past_key_values 的实现见: transformers.cache_utils.DynamicCache
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping.get(decoder_layer.attention_type, causal_mask_mapping["full_attention"]),
            position_ids=text_position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )




# =================================================================================================
# ================================ MODIFIED BY GEMINI: START ======================================
# =================================================================================================
# video_start_id = config.vision_start_token_id
# video_end_id = config.vision_end_token_id
# skeleton_start_id = config.skeleton_config['skeleton_start_token_id']
# skeleton_end_id = config.skeleton_config['skeleton_end_token_id']
def create_custom_pose_attention_mask_baseV2(
    input_ids: torch.Tensor,
    base_causal_mask: torch.Tensor,
    config: Qwen2_5_VLConfig,
    is_training: bool,
) -> torch.Tensor:
    """
    【最终版】根据训练或推理范式，创建用于视频姿态估计的、分块的布尔型注意力遮罩。

    此函数能够精确处理文本、视频、骨架区域交错的复杂模板，并根据当前模式
    (训练或推理) 应用正确的注意力规则。

    - 训练模式 (is_training=True): 
      为骨架token启用全连接注意力，以最高效地学习姿态的全局空间结构。

    - 推理模式 (is_training=False): 
      为骨架token强制使用因果注意力，以保证自回归生成过程的逻辑正确性。
    
    Args:
        input_ids (torch.Tensor): 输入的 token ID 序列, shape [B, L].
        base_causal_mask (torch.Tensor): Hugging Face 生成的基础因果遮罩 (下三角布尔矩阵).
        config (Qwen2_5_VLConfig): 模型配置，用于获取特殊 token 的 ID。
        is_training (bool): 当前是否处于训练模式。
        
    Returns:
        torch.Tensor: 最终的、符合当前范式需求的注意力遮罩。
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # 步骤 1: 从配置中安全地获取所有需要的特殊 token ID
    try:
        # 注意：请再次确认这些 key 在您的 config 对象中路径正确
        video_start_id = config.vision_start_token_id
        video_end_id = config.vision_end_token_id
        skeleton_start_id = config.skeleton_config['skeleton_start_token_id']
        skeleton_end_id = config.skeleton_config['skeleton_end_token_id']
    except (AttributeError, KeyError) as e:
        # 如果配置不完整或任务不相关，打印警告并回退到原始的因果遮罩，增加代码稳健性
        logger.warning_once(f"Could not find required special token IDs for custom attention mask ({e}). Falling back to causal mask.")
        return base_causal_mask

    # 步骤 2: 精确识别视频和骨架区域
    # 初始化所有区域mask为False
    video_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    skeleton_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    # 遍历batch中的每个样本，独立地识别其区域块
    for b in range(batch_size):
        ids = input_ids[b]
        
        # 识别所有视频块 (start...end)
        vid_start_indices = (ids == video_start_id).nonzero(as_tuple=True)[0]
        vid_end_indices = (ids == video_end_id).nonzero(as_tuple=True)[0]
        for start, end in zip(vid_start_indices, vid_end_indices):
            if start < end:
                video_mask[b, start:end + 1] = True
            
        # 识别所有骨架块 (start...end)
        skel_start_indices = (ids == skeleton_start_id).nonzero(as_tuple=True)[0]
        skel_end_indices = (ids == skeleton_end_id).nonzero(as_tuple=True)[0]
        for start, end in zip(skel_start_indices, skel_end_indices):
            if start < end:
                skeleton_mask[b, start:end + 1] = True

    # 步骤 3: 识别文本区域
    # 文本区域 = 非视频区域 AND 非骨架区域。这种定义方法非常稳健。
    text_mask = ~(video_mask | skeleton_mask)

    # 步骤 4: 构建两种范式共通的注意力规则
    # 规则 A: Video Query -> (Text | Video) Key
    video_to_text_video_mask = torch.einsum('bq,bk->bqk', video_mask, (text_mask | video_mask))
    
    # 规则 B: Skeleton Query -> (Text | Video) Key
    skeleton_to_text_video_mask = torch.einsum('bq,bk->bqk', skeleton_mask, (text_mask | video_mask))

    # 将这些共通规则应用到基础因果遮罩上。
    # .squeeze(1) 是因为 base_causal_mask 的形状是 [B, 1, L, L]，需要与 [B, L, L] 的区域mask对齐
    # .clone() 确保我们不会在原地修改原始mask，避免潜在的副作用
    final_mask = base_causal_mask.squeeze(1).clone() | video_to_text_video_mask | skeleton_to_text_video_mask

    # 步骤 5: 应用范式特定的规则 (这是整个逻辑的核心)
    if is_training:
        # --- 训练范式 ---
        # 目标: 学习全局结构
        # 行为: 打开 Skeleton -> Skeleton 的全连接注意力，允许模型看到完整的骨架结构。
        
        skeleton_to_skeleton_mask = torch.einsum('bq,bk->bqk', skeleton_mask, skeleton_mask)
        final_mask = final_mask | skeleton_to_skeleton_mask
    
    # --- 推理范式 ---
    # 在 is_training=False 的情况下，我们不执行任何额外操作。
    # final_mask 中 `Skeleton -> Skeleton` 的注意力部分继承自 `base_causal_mask`，
    # 它本身就是因果的。这优雅地满足了自回归生成的要求。

    # 返回前，将mask的形状恢复为 [B, 1, L, L] 以匹配注意力层的期望输入
    return final_mask.unsqueeze(1)
# =================================================================================================
# ================================= MODIFIED BY GEMINI: END =======================================
# =================================================================================================


def create_custom_pose_attention_mask_nar(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    base_causal_mask: torch.Tensor,
    config: Qwen2_5_VLConfig,
    is_training: bool,
) -> torch.Tensor:
    if base_causal_mask is None:
        return None

    # ================================ MODIFIED BY GEMINI: START ================================
    # 关键修复:
    # query_seq_len 来自 4D mask 的第3维
    # key_seq_len 来自 2D attention_mask 的第2维 (最可靠的完整长度来源)
    batch_size, _, query_seq_len, _ = base_causal_mask.shape
    _, key_seq_len = attention_mask.shape
    device = input_ids.device
    # ================================ MODIFIED BY GEMINI: END ==================================

    try:
        video_start_id = config.vision_start_token_id
        video_end_id = config.vision_end_token_id
        skeleton_start_id = config.skeleton_config['skeleton_start_token_id']
        skeleton_end_id = config.skeleton_config['skeleton_end_token_id']
    except (AttributeError, KeyError) as e:
        logger.warning_once(f"Could not find required special token IDs ({e}). Falling back to causal mask.")
        return base_causal_mask

    final_mask = base_causal_mask.squeeze(1).clone()

    # 使用正确的 key_seq_len 初始化所有基于完整序列的 1D masks
    full_video_mask = torch.zeros((batch_size, key_seq_len), dtype=torch.bool, device=device)
    full_skeleton_mask = torch.zeros((batch_size, key_seq_len), dtype=torch.bool, device=device)

    # 这里的 input_ids 是我们传入的 full_input_ids_for_mask，长度与 key_seq_len 一致
    for b in range(batch_size):
        ids = input_ids[b]
        
        vid_start_indices = (ids == video_start_id).nonzero(as_tuple=True)[0]
        vid_end_indices = (ids == video_end_id).nonzero(as_tuple=True)[0]
        for start, end in zip(vid_start_indices, vid_end_indices):
            if start < end:
                full_video_mask[b, start:end + 1] = True
        
        # 训练时，骨架区域由 start/end token 识别
        if is_training:
            skel_start_indices = (ids == skeleton_start_id).nonzero(as_tuple=True)[0]
            skel_end_indices = (ids == skeleton_end_id).nonzero(as_tuple=True)[0]
            for start, end in zip(skel_start_indices, skel_end_indices):
                if start < end:
                    full_skeleton_mask[b, start:end + 1] = True

    full_text_mask = ~(full_video_mask | full_skeleton_mask)

    if is_training:
        # --- 训练范式 (L_q == L_k) ---
        visible_area = full_text_mask | full_video_mask | full_skeleton_mask
        update_mask = visible_area.unsqueeze(1).expand(-1, query_seq_len, -1)
        
        is_query_mask = full_skeleton_mask.unsqueeze(2)
        
        final_mask = torch.where(is_query_mask, update_mask, final_mask)
    
    else:
        # --- 推理范式 ---
        pose_zone_in_keys = (attention_mask == 2)
        
        if torch.any(pose_zone_in_keys):
            # --- NAR 步骤 (L_q != L_k) ---
            visible_area = full_text_mask | full_video_mask | pose_zone_in_keys
            update_mask = visible_area.unsqueeze(1).expand(-1, query_seq_len, -1)

            is_query_mask = pose_zone_in_keys.unsqueeze(2)
        
            final_mask = torch.where(is_query_mask, update_mask, final_mask)

    # import matplotlib.pyplot as plt
    # plt.imsave('tmp.png', final_mask[0].cpu().int().numpy())
    return final_mask.unsqueeze(1)