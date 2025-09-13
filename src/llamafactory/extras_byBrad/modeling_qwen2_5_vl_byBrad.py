import os
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
    Qwen2_5_VLModel, Qwen2_5_VLModelOutputWithPast, Qwen2_5_VLCausalLMOutputWithPast,
    is_torchdynamo_compiling,
)
from transformers.utils import can_return_tuple, auto_docstring
from transformers.loss.loss_utils import fixed_cross_entropy, ForCausalLMLoss
from safetensors.torch import load_file
from llamafactory.extras_byBrad.vqvae import SKEL_VQVAE as SkeletonProcessor, Encoder, VectorQuantizer, Decoder


class Qwen2_5_VLForConditionalGenerationWithSkeleton(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config: Qwen2_5_VLConfig, **kwargs_byBrad):
        super().__init__(config)

        ########## MPJPE EXRTA LOSS PART #####################################################################################################################   
        self.use_mpjpe_loss = kwargs_byBrad['use_mpjpe_loss']
        self.mpjpe_success_count = 0

        ########## VQVAE PART #####################################################################################################################   
        self.vqvae_ckpt = kwargs_byBrad['vqvae_ckpt']
        
        encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1])
        vq = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)
        decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
        self._skeleton_processor_container = [SkeletonProcessor(encoder, decoder, vq)]
        for param in self._skeleton_processor_container[0].parameters():
            param.requires_grad = False
        self._skeleton_processor_container[0].eval()

        self.is_vqvae_weights_loaded = False

        ########## FORWARD PART#####################################################################################################################
        self.model.forward = types.MethodType(custom_qwen2_5_vl_model_forward, self.model)

    def load_vqvae_weights(self):
        """
        加载 VQ-VAE 的权重到正确的设备上。
        """
        # 获取 skeleton_processor 当前所在的设备
        device = next(self.skeleton_processor.parameters()).device
        print(f"Loading VQ-VAE weights from {self.vqvae_ckpt} to device: {device}")
        
        # 从文件加载权重，直接加载到目标设备
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
                new_processor = type(processor)(
                    encoder=type(processor.encoder)(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1]),
                    decoder=type(processor.decoder)(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0]),
                    vq=type(processor.vq)(nb_code=8192, code_dim=3072, is_train=False),
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

                    # 从 logits 中获取预测的 token ID (不带梯度)
                    pred_ids = torch.argmax(logits.detach(), dim=-1)    # [1,352]

                    # 步骤 1: 识别真实 `labels` 中的骨架 token 区域
                    # is_gt_skeleton_mask: [B, L], bool, 标记哪些位置应该是骨架 token
                    is_gt_skeleton_mask = torch.isin(labels, self._skeleton_token_id_tensor)    # [1,352]

                    # 如果当前批次中没有任何骨架 token，则直接跳过
                    if torch.any(is_gt_skeleton_mask):
                        # 步骤 2: 获取模型在这些真实骨架区域的预测
                        # pred_ids_in_gt_skel_region: [N], N 是骨架 token 的总数
                        pred_ids_in_gt_skel_region = pred_ids[is_gt_skeleton_mask]

                        # 步骤 3: 验证所有这些预测是否也都是有效的骨架 token
                        # all_preds_are_valid: bool, 检查是否所有预测都在骨架 token ID 集合中
                        all_preds_are_valid = torch.isin(pred_ids_in_gt_skel_region, self._skeleton_token_id_tensor).all()

                        # 步骤 4: 如果所有预测都有效，才计算 MPJPE
                        if all_preds_are_valid:
                            # 从有效的预测 token ID 中解析出 VQ-VAE 索引
                            # 注意：这里我们直接使用 pred_ids_in_gt_skel_region，因为它已经被验证过
                            pred_skel_indices = self._parse_skeleton_indices_from_ids(pred_ids_in_gt_skel_region)

                            if pred_skel_indices.numel() > 0:
                                # 解码预测的索引以获得 3D 姿态
                                pred_poses = self.skeleton_processor.decode(pred_skel_indices)

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
        input_ids=None,
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