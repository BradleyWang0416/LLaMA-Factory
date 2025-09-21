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

logger = logging.get_logger(__name__)


class Qwen2_5_VLForConditionalGenerationWithSkeleton(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config: Qwen2_5_VLConfig, **kwargs_byBrad):
        super().__init__(config)

        ########## SKELETON ATTENTION PART #####################################################################################################################
        self.skeleton_attention_type = kwargs_byBrad['skeleton_attention_type']
        if self.skeleton_attention_type is not None:
            assert self.skeleton_attention_type in ['base', 'base_v2']
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
        else:
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
        # --------------------
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # MODIFIED BY GEMINI: 移除了在这里创建 skeleton_mask 的逻辑。
        # 它将在 custom_qwen2_5_vltextmodel_forward 中被统一创建和处理。
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






# =================================================================================================
# ================================ MODIFIED BY GEMINI: START ======================================
# =================================================================================================


# video_start_id = config.vision_start_token_id
# video_end_id = config.vision_end_token_id
# skeleton_start_id = config.skeleton_config['skeleton_start_token_id']
# skeleton_end_id = config.skeleton_config['skeleton_end_token_id']
def create_custom_pose_attention_mask(
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
        final_mask = create_custom_pose_attention_mask(
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