import torch
from typing import Optional
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

def get_rope_index(
    self: Qwen2_5_VLModel,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    skeleton_grid_thw: Optional[torch.LongTensor] = None, # ADDED FOR SKELETON
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image, video, and skeleton's temporal, height and width in LLM.
    ... (docstring remains the same, but should be updated to include skeleton) ...
    """
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    # --- ADDED FOR SKELETON ---
    skeleton_token_id = self.config.skeleton_token_id   # 在 src/llamafactory/model/loader.py 的 load_model 中添加
    vision_start_token_id = self.config.vision_start_token_id
    # Assuming a similar start token for skeletons, or it might be the same as vision_start
    skeleton_start_token_id = self.config.skeleton_start_token_id   # 在 src/llamafactory/model/loader.py 的 load_model 中添加
    # --- END ADDED ---

    mrope_position_deltas = []
    # --- MODIFIED: Check for any multimodal input ---
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None or skeleton_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index, skeleton_index = 0, 0, 0 # ADDED skeleton_index
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, current_input_ids_sample in enumerate(total_input_ids):
            input_ids_unpadded = current_input_ids_sample[attention_mask[i] == 1]
            
            # --- MODIFIED: Find all vision/skeleton start indices ---
            vision_start_indices = torch.argwhere(input_ids_unpadded == vision_start_token_id).squeeze(1)
            skeleton_start_indices = torch.argwhere(input_ids_unpadded == skeleton_start_token_id).squeeze(1)
            
            # Identify the type of token following each start token
            vision_tokens = input_ids_unpadded[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()

            skeleton_tokens = input_ids_unpadded[skeleton_start_indices + 1]
            skeleton_nums = (skeleton_tokens == skeleton_token_id).sum()
            # --- END MODIFIED ---

            input_tokens = input_ids_unpadded.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_skeletons = image_nums, video_nums, skeleton_nums

            # --- MODIFIED: Main loop to process all modalities ---
            for _ in range(image_nums + video_nums + skeleton_nums):
                # Find the next occurrence of each modality token
                try:
                    ed_image = input_tokens.index(image_token_id, st) if remain_images > 0 else float('inf')
                except ValueError:
                    ed_image = float('inf')
                
                try:
                    ed_video = input_tokens.index(video_token_id, st) if remain_videos > 0 else float('inf')
                except ValueError:
                    ed_video = float('inf')

                try:
                    ed_skeleton = input_tokens.index(skeleton_token_id, st) if remain_skeletons > 0 else float('inf')
                except ValueError:
                    ed_skeleton = float('inf')

                # Determine which modality comes first
                min_ed = min(ed_image, ed_video, ed_skeleton)
                
                if min_ed == float('inf'): # Should not happen if counts are correct
                    break

                if min_ed == ed_image:
                    t, h, w = image_grid_thw[image_index]
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                elif min_ed == ed_video:
                    t, h, w = video_grid_thw[video_index]
                    second_per_grid_t = second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                else: # Skeleton case. No spatial merge
                    t, h, w = skeleton_grid_thw[skeleton_index]
                    second_per_grid_t = 0 # Skeletons are like images, no time progression
                    skeleton_index += 1
                    remain_skeletons -= 1
                    ed = ed_skeleton
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item(),   # <-- No division by spatial_merge_size. 与 src/llamafactory/data/mm_plugin.py::Qwen2VLPlugin::process_messages 的逻辑保持一致
                        w.item()    # <-- No division by spatial_merge_size
                    )
                # --- END MODIFIED ---

                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                second_per_grid_t = torch.as_tensor(
                    second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                )

                time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # Fallback for pure text or no multimodal inputs
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas