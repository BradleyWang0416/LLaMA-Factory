import os
import sys
from typing import Optional, Union, Tuple
import types
import torch
import torch.nn as nn
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

class Qwen2_5_VLForConditionalGenerationWithSkeleton(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config: Qwen2_5_VLConfig, **kwargs_byBrad):
        super().__init__(config)

        ########## SKELETON ATTENTION PART #####################################################################################################################
        self.skeleton_attention_type = kwargs_byBrad['skeleton_attention_type']
        if self.skeleton_attention_type is not None:
            assert self.skeleton_attention_type in ['base', 'base_v2', 'nar', 'deformable_attn_w_joint2dcpn']
        setattr(self.model, 'skeleton_attention_type', self.skeleton_attention_type)
        setattr(self.model.language_model, 'skeleton_attention_type', self.skeleton_attention_type)


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
        skeleton_indices: Optional[torch.LongTensor] = None,  # 骨架数据的索引
        skeleton_poses: Optional[torch.FloatTensor] = None,  # 骨架数据的3D姿态
        skeleton_grid_thw: Optional[torch.LongTensor] = None,  # 骨架数据的网格尺寸
        source_slice_id = None,  # 用于解码的源切片ID
        joint2d_cpn_affined_normed = None,
        # --------------------
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:



        if self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn':
            assert joint2d_cpn_affined_normed is not None
            try:
                joint2d_cpn_affined_normed = torch.stack(joint2d_cpn_affined_normed)
            except:
                pass
            kwargs['joint2d_cpn_affined_normed'] = joint2d_cpn_affined_normed


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

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
        


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

        if self.skeleton_attention_type != 'nar':
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

    # <<< 新增/重载方法 2: 核心生成循环 >>>
    # 这是对 _sample 方法的重载，以实现我们的混合生成逻辑
# <<< 新增/重载方法 2: 核心生成循环 (已由Gemini修复) >>>
    # 这是对 _sample 方法的重载，以实现我们的混合生成逻辑
    def _sample_v0(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        generation_config: Optional[GenerationConfig] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

        # <<< MODIFIED FOR AR-NAR >>>
        # 步骤 1: 添加卫语句，如果不是 'nar' 模式，则直接调用并返回父类的标准方法
        if self.skeleton_attention_type != 'nar':
            # 为了确保与官方版本完全一致，我们直接调用 GenerationMixin 的 _sample 方法
            # 而不是 Qwen 自身的 super()._sample()，因为 Qwen 可能没有重载它
            return GenerationMixin._sample(
                self,
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        
        # --- 步骤 2: 标准初始化 (来自官方源码) ---
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample
        
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # ================================ MODIFIED BY GEMINI: START ================================
        #                      --- 步骤 3: 正确的初始缓存位置处理 (关键修复) ---
        # 调用官方辅助函数来正确初始化 cache_position，这解决了第一次循环的错误。
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        # ================================ MODIFIED BY GEMINI: END ==================================


        # <<< MODIFIED FOR AR-NAR >>>
        # 步骤 3: 添加我们自己的 AR-NAR 专用初始化
        skel_start_id = self.config.skeleton_config['skeleton_start_token_id']
        skel_end_id = self.config.skeleton_config['skeleton_end_token_id']
        query_token_ids = self.config.skeleton_config.get('skeleton_query_token_indices')
        if query_token_ids is None:
            raise ValueError("`query_token_ids` must be defined in `config.skeleton_config` for NAR generation.")
        num_pose_tokens = len(query_token_ids)
        query_tokens_tensor = torch.tensor(query_token_ids, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        
        # 核心状态标志，现在通过 model_kwargs 传递，以兼容 torch.compile
        model_kwargs["next_is_pose_block"] = False

        # --- 步骤 4: 标准的预处理和编译逻辑 (来自官方源码) ---
        if generation_config.prefill_chunk_size is not None:
            # 这个逻辑块处理长输入的预填充优化
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        # --- 步骤 5: 核心生成循环 (融合了AR-NAR逻辑) ---
        iter_cnt = 0
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):            
            # <<< MODIFIED FOR AR-NAR >>>
            # 检查是否正要进入NAR步骤
            is_entering_nar_step = model_kwargs.pop("next_is_pose_block", False)
            if is_entering_nar_step:
                input_ids = torch.cat([input_ids, query_tokens_tensor], dim=-1)
                attention_mask = model_kwargs.get("attention_mask")
                pose_mask_signal = torch.full(
                    (attention_mask.shape[0], num_pose_tokens), 2,
                    device=attention_mask.device, dtype=attention_mask.dtype
                )
                model_kwargs['attention_mask'] = torch.cat([attention_mask, pose_mask_signal], dim=-1)
                
                past_key_values = model_kwargs.get("past_key_values")
                if past_key_values is not None:
                    past_length = past_key_values[0][0].shape[2]
                    model_kwargs['cache_position'] = torch.arange(
                        past_length, past_length + num_pose_tokens, device=input_ids.device
                    )

            # ================================ MODIFIED BY GEMINI: START ================================
            # 关键修复: 在调用 prepare_inputs_for_generation 之前，
            # 将完整的 input_ids 存入 kwargs，以供下游的自定义 attention mask 函数使用。
            model_kwargs["full_input_ids_for_mask"] = input_ids
            # ================================ MODIFIED BY GEMINI: END ==================================

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # 前向传播
            outputs = self(**model_inputs, return_dict=True)

            # 更新KV缓存等
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # <<< MODIFIED FOR AR-NAR >>>
            # 步骤 6: 根据AR或NAR模式，条件性地提取logits
            if is_entering_nar_step:
                # --- NAR 模式 ---
                next_token_logits = outputs.logits[:, -num_pose_tokens:, :]
            else:
                # --- AR 模式 ---
                next_token_logits = outputs.logits[:, -1, :]
            
            # (后续的 logits 处理和解码逻辑与您之前的版本相同)
            if next_token_logits.ndim == 3: # NAR
                next_token_scores = next_token_logits
            else: # AR
                next_token_scores = logits_processor(input_ids, next_token_logits)

            # --- 步骤 7: 标准的Token选择逻辑 (来自官方源码，但我们处理多token情况) ---
            if do_sample:
                probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                probs = probs.view(input_ids.shape[0], -1, probs.shape[-1])
                next_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(probs.shape[0], probs.shape[1])
            else:
                # 对于NAR，我们通常也使用采样，但这里为了完整性保留argmax
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # <<< MODIFIED FOR AR-NAR >>>
            # 步骤 8: 根据AR或NAR模式，条件性地更新 input_ids 和状态
            next_is_pose_block_for_next_iter = False
            assert next_tokens.numel() == 1
            if next_tokens.shape[1] == 1 and next_tokens.item() == skel_start_id:
                next_is_pose_block_for_next_iter = True
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            elif next_tokens.shape[1] == num_pose_tokens:
                input_ids[:, -num_pose_tokens:] = next_tokens
                end_token = torch.full((input_ids.shape[0], 1), skel_end_id, device=input_ids.device, dtype=torch.long)
                input_ids = torch.cat([input_ids, end_token], dim=-1)
            else:
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # 将下一轮的状态标志存入 model_kwargs
            model_kwargs["next_is_pose_block"] = next_is_pose_block_for_next_iter
            
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # --- 步骤 9: 标准的停止条件检查 (来自官方源码, 已融合我们的修复) ---
            # 只在AR步骤检查EOS
            if next_tokens.shape[1] == 1:
                if next_tokens[0].item() in eos_token_id:
                    unfinished_sequences.fill_(0)
            
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            
            if this_peer_finished and not synced_gpus:
                break

            iter_cnt += 1
        
        # --- 步骤 10: 标准的返回逻辑 (来自官方源码) ---
        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=None, # 您可以按需填充
                hidden_states=None,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids
    
    def _sample_v1(
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
                
                # 扩展 attention_mask
                attention_mask = model_kwargs.get("attention_mask")
                pose_mask_signal = torch.full(
                    (attention_mask.shape[0], num_pose_tokens), 2,
                    device=attention_mask.device, dtype=attention_mask.dtype
                )
                model_kwargs['attention_mask'] = torch.cat([attention_mask, pose_mask_signal], dim=-1)

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
        

        if self.skeleton_attention_type == 'deformable_attn_w_joint2dcpn':
            joint2d_cpn_affined_normed = kwargs['joint2d_cpn_affined_normed']

            assert (video_mask[:,:,:1] == video_mask[:,:,1:]).all()
            video_mask_unpad = video_mask[:,:,0]    # [1,352]
            kwargs['video_mask_unpad'] = video_mask_unpad




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
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
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




def create_custom_pose_attention_mask_nar_v0(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    base_causal_mask: torch.Tensor,
    config: Qwen2_5_VLConfig,
    is_training: bool,
) -> torch.Tensor:
    """
    【最终修正版】根据训练、推理或特定信号，创建分块的布尔型注意力遮罩。
    此版本已修复在解码阶段因 query_length != key_length 导致的形状不匹配错误。
    """
    # 在解码步骤中, base_causal_mask 可能为 None, 此时直接返回
    if base_causal_mask is None:
        return None

    # --- 步骤 1: 获取形状和设备信息 ---
    # base_causal_mask 形状为 [B, 1, L_q, L_k]
    batch_size, _, query_seq_len, key_seq_len = base_causal_mask.shape
    device = input_ids.device

    # --- 步骤 2: 从配置中安全地获取特殊 token ID ---
    try:
        video_start_id = config.vision_start_token_id
        video_end_id = config.vision_end_token_id
        skeleton_start_id = config.skeleton_config['skeleton_start_token_id']
        skeleton_end_id = config.skeleton_config['skeleton_end_token_id']
    except (AttributeError, KeyError) as e:
        logger.warning_once(f"Could not find required special token IDs ({e}). Falling back to causal mask.")
        return base_causal_mask

    # --- 步骤 3: 准备基础遮罩 ---
    final_mask = base_causal_mask.squeeze(1).clone() # 形状: [B, L_q, L_k]

    # --- 步骤 4: 构建作用于完整 Key 序列 (长度 L_k) 的一维区域遮罩 ---
    # 这里的 input_ids 是完整的序列 (full_input_ids_for_mask)
    full_video_mask = torch.zeros((batch_size, key_seq_len), dtype=torch.bool, device=device)
    full_skeleton_mask = torch.zeros((batch_size, key_seq_len), dtype=torch.bool, device=device)

    for b in range(batch_size):
        ids = input_ids[b]
        
        vid_start_indices = (ids == video_start_id).nonzero(as_tuple=True)[0]
        vid_end_indices = (ids == video_end_id).nonzero(as_tuple=True)[0]
        for start, end in zip(vid_start_indices, vid_end_indices):
            if start < end:
                full_video_mask[b, start:end + 1] = True
        
        skel_start_indices = (ids == skeleton_start_id).nonzero(as_tuple=True)[0]
        skel_end_indices = (ids == skeleton_end_id).nonzero(as_tuple=True)[0]
        for start, end in zip(skel_start_indices, skel_end_indices):
            if start < end:
                full_skeleton_mask[b, start:end + 1] = True

    full_text_mask = ~(full_video_mask | full_skeleton_mask)

    # --- 步骤 5: 应用范式特定的规则 (核心修复逻辑) ---
    if is_training:
        # --- 训练范式 (L_q == L_k) ---
        # 允许骨架token之间，以及骨架到(文本+视频)的注意力
        visible_area = full_text_mask | full_video_mask | full_skeleton_mask
        # unsqueeze(1) 将 [B, L_k] -> [B, 1, L_k]
        # expand 将其复制 L_q 次 -> [B, L_q, L_k]
        update_mask = visible_area.unsqueeze(1).expand(-1, query_seq_len, -1)
        
        # is_query_mask: 标记哪些 Query token (行) 需要应用这个规则
        is_query_mask = full_skeleton_mask.unsqueeze(2) # [B, L_q, 1]
        
        # 只在骨架token作为查询时，才应用这个全连接的遮罩
        final_mask = torch.where(is_query_mask, update_mask, final_mask)
    
    else:
        # --- 推理范式 ---
        # 检查 attention_mask 中是否有特殊信号 (值为2)
        # 注意: 这里的 attention_mask 是完整的 2D mask
        pose_zone_in_keys = (attention_mask == 2) # 形状: [B, L_k]
        
        if torch.any(pose_zone_in_keys):
            # --- NAR 步骤 (L_q != L_k) ---
            # 1. 定义 pose queries 允许看到的所有区域
            visible_area = full_text_mask | full_video_mask | pose_zone_in_keys
            
            # 2. 将这个 [B, L_k] 的可视区域遮罩广播到 [B, L_q, L_k]
            update_mask = visible_area.unsqueeze(1).expand(-1, query_seq_len, -1)
            
            # 3. 将这个更新应用到 final_mask 上
            # 在NAR步骤, 所有的 query (L_q=68) 都需要应用这个规则, 所以直接合并
            final_mask = final_mask | update_mask
        
        # --- AR 步骤 ---
        # 如果没有信号，则不执行任何操作，直接返回继承自 base_causal_mask 的因果遮罩

    # 返回前，将mask的形状恢复为 [B, 1, L_q, L_k]
    return final_mask.unsqueeze(1)



def create_custom_pose_attention_mask_nar_v0(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    base_causal_mask: torch.Tensor,
    config: Qwen2_5_VLConfig,
    is_training: bool,
) -> torch.Tensor:
    """
    【最终修正版-2】根据训练、推理或特定信号，创建分块的注意力遮罩。
    此版本修复了 key_length 的推断错误。
    """
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


            # ================================ MODIFIED BY GEMINI: START ================================
            #                      --- 【最终BUG修复】手动校正 final_mask 的形状 ---
            
            # 检查并修复 final_mask 和 update_mask 之间的 "差一" 问题
            if final_mask.shape[2] < update_mask.shape[2]:
                padding_needed = update_mask.shape[2] - final_mask.shape[2]
                # 使用 False (0) 进行填充，因为对于因果遮罩，默认是不可见的
                padding = torch.zeros(
                    final_mask.shape[0], final_mask.shape[1], padding_needed, 
                    dtype=torch.bool, device=final_mask.device
                )
                final_mask = torch.cat([final_mask, padding], dim=2)
            
            # ================================ MODIFIED BY GEMINI: END ==================================

            final_mask = final_mask | update_mask

    # import matplotlib.pyplot as plt
    # plt.imsave('tmp.png', final_mask[0].cpu().int().numpy())
    return final_mask.unsqueeze(1)



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