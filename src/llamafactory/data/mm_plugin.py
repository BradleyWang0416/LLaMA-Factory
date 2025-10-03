# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/processing_llava.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Literal, Optional, TypedDict, Union
import json
import joblib
from collections import defaultdict
from time import time

import numpy as np
import torch
from transformers.image_utils import get_image_size, is_valid_image, to_numpy_array
from transformers.models.mllama.processing_mllama import (
    convert_sparse_cross_attention_mask_to_dense,
    get_cross_attention_token_mask,
)
from typing_extensions import override

from ..extras.constants import AUDIO_PLACEHOLDER, IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER

# ADDED BY BRADLEY 250827 ###############################################################
from ..extras.constants import SKELETON_PLACEHOLDER, SKELETON_TOKEN_BASE, SKELETON_FRAME_BREAK
from ..extras_byBrad.vqvae import SKEL_VQVAE as SkeletonProcessor
#########################################################################################
# ADDED BY BRADLEY 250906 ###############################################################
from ..extras.constants import BODY_PART_TOKENS, JOINT_GROUP_MAP, BODY_PART_ORDER
from ..extras_byBrad.convert_skel_token import *
#########################################################################################
# ADDED BY BRADLEY 251002 ###############################################################
from ..extras.constants import PROMPT_PLACEHOLDER, RESPONSE_PLACEHOLDER
from _llamafactory_skeleton_byBrad.data_utils.templates import PROMPT_TEMPLATES
import _llamafactory_skeleton_byBrad.data_utils.convert_skel_token as convert_skel_token_v2
import sys
sys.path.append("../ContextAwarePoseFormer_Private/H36M-Toolbox/")
from multimodal_h36m_dataset_byBradley import Multimodal_Mocap_Dataset, DATA_ROOT_PATH
sys.path.remove("../ContextAwarePoseFormer_Private/H36M-Toolbox/")
#########################################################################################

from ..extras.packages import (
    is_librosa_available,
    is_pillow_available,
    is_pyav_available,
    is_transformers_version_greater_than,
)


if is_librosa_available():
    import librosa


if is_pillow_available():
    from PIL import Image
    from PIL.Image import Image as ImageObject


if is_pyav_available():
    import av


if is_transformers_version_greater_than("4.52.0"):
    from transformers.image_utils import make_flat_list_of_images
    from transformers.video_utils import make_batched_videos
else:
    from transformers.image_utils import make_batched_videos, make_flat_list_of_images


if TYPE_CHECKING:
    from av.stream import Stream
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
    VideoInput = Union[str, BinaryIO, list[list[ImageInput]]]
    AudioInput = Union[str, BinaryIO, NDArray]

    # ADDED BY BRADLEY 250827 ###############################################################
    SkeletalInput = Union[str, BinaryIO, NDArray]
    #########################################################################################

    class MMProcessor(ProcessorMixin):
        patch_size: int
        image_seq_length: int
        num_additional_image_tokens: int
        vision_feature_select_strategy: Literal["default", "full"]

        def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
            pass



def _make_batched_images(images: list["ImageObject"], imglens: list[int]) -> list[list["ImageObject"]]:
    r"""Make nested list of images."""
    batch_images = []
    for imglen in imglens:
        batch_images.append(images[:imglen])
        images = images[imglen:]

    return batch_images


def _check_video_is_nested_images(video: "VideoInput") -> bool:
    r"""Check if the video is nested images."""
    return isinstance(video, list) and all(isinstance(frame, (str, BinaryIO, dict)) for frame in video)


@dataclass
class MMPluginMixin:
    image_token: Optional[str]
    video_token: Optional[str]
    audio_token: Optional[str]
    skeleton_token: Optional[str] # ADDED BY BRADLEY 250827
    expand_mm_tokens: bool = True

    def _validate_input(
        self,
        processor: Optional["MMProcessor"],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        skeletons: list["SkeletalInput"], # ADDED BY BRADLEY 250827
    ) -> None:
        r"""Validate if this model accepts the input modalities."""
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(
            processor, "video_processor", getattr(processor, "image_processor", None)
        )
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

        if len(audios) != 0 and self.audio_token is None:
            raise ValueError(
                "This model does not support audio input. Please check whether the correct `template` is used."
            )
        
        # ADDED BY BRADLEY 250827 ###############################################################
        # if len(skeletons) != 0 and self.skeleton_token is None:
        #     raise ValueError(
        #         "This model does not support skeleton input. Please check whether the correct `template` is used."
        #     )
        #########################################################################################

        if self.image_token is not None and processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")

        if self.image_token is not None and image_processor is None:
            raise ValueError("Image processor was not found, please check and update your model file.")

        if self.video_token is not None and video_processor is None:
            raise ValueError("Video processor was not found, please check and update your model file.")

        if self.audio_token is not None and feature_extractor is None:
            raise ValueError("Audio feature extractor was not found, please check and update your model file.")
        

    def _validate_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        skeletons: list["SkeletalInput"], # ADDED BY BRADLEY 250827
    ):
        r"""Validate if the number of images, videos and audios match the number of placeholders in messages."""
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        num_skeleton_tokens = 0 # ADDED BY BRADLEY 250827
        for message in messages:
            num_image_tokens += message["content"].count(IMAGE_PLACEHOLDER)
            num_video_tokens += message["content"].count(VIDEO_PLACEHOLDER)
            num_audio_tokens += message["content"].count(AUDIO_PLACEHOLDER)
            num_skeleton_tokens += message["content"].count(SKELETON_PLACEHOLDER) # ADDED BY BRADLEY 250827

        # DELETED BY BRADLEY 251002 ###############################################################
        # if len(images) != num_image_tokens:
        #     raise ValueError(
        #         f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens in {messages}."
        #     )

        # if len(videos) != num_video_tokens:
        #     raise ValueError(
        #         f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens in {messages}."
        #     )
        #########################################################################################

        if len(audios) != num_audio_tokens:
            raise ValueError(
                f"The number of audios does not match the number of {AUDIO_PLACEHOLDER} tokens in {messages}."
            )
        # ADDED BY BRADLEY 250827 ###############################################################
        if len(skeletons) != num_skeleton_tokens:
            raise ValueError(
                f"The number of skeletons does not match the number of {SKELETON_PLACEHOLDER} tokens in {messages}."
            )
        #########################################################################################

    def _preprocess_image(
        self, image: "ImageObject", image_max_pixels: int, image_min_pixels: int, **kwargs
    ) -> "ImageObject":
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_indices(
        self, video_stream: "Stream", video_fps: float, video_maxlen: int, **kwargs
    ) -> list[int]:
        r"""Compute video sample indices according to fps."""
        total_frames = video_stream.frames
        if total_frames == 0:  # infinite video
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        sample_frames = max(1, math.floor(float(video_stream.duration * video_stream.time_base) * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)

    def _regularize_images(self, images: list["ImageInput"], **kwargs) -> dict[str, list["ImageObject"]]:
        r"""Regularize images to avoid error. Including reading and pre-processing."""
        results = []
        for image in images:
            if isinstance(image, (str, BinaryIO)):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return {"images": results}

    def _regularize_videos(self, videos: list["VideoInput"], **kwargs) -> dict[str, list[list["ImageObject"]]]:
        r"""Regularizes videos to avoid error. Including reading, resizing and converting."""
        results = []
        for video in videos:
            frames: list[ImageObject] = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")
                frames = video
            else:
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {"videos": results}

    def _regularize_audios(
        self, audios: list["AudioInput"], sampling_rate: float, **kwargs
    ) -> dict[str, Union[list["NDArray"], list[float]]]:
        r"""Regularizes audios to avoid error. Including reading and resampling."""
        results, sampling_rates = [], []
        for audio in audios:
            if not isinstance(audio, np.ndarray):
                audio, sampling_rate = librosa.load(audio, sr=sampling_rate)

            results.append(audio)
            sampling_rates.append(sampling_rate)

        return {"audios": results, "sampling_rates": sampling_rates}


    # ADDED BY BRADLEY 250827 ###############################################################
    def _regularize_skeletons(
        self, skeletons: list["SkeletalInput"], **kwargs
    ) -> dict[str, "torch.Tensor"]:
        r"""Regularizes skeletons by encoding with VQVAE."""

        if isinstance(skeletons[0], str) and skeletons[0].endswith('.npy') and 'skeleton_code' in skeletons[0]:
            POSES = []
            CODEBOOK_INDICES = []
            GRID_SHAPES = []
            SOURCE_SLICE_ID = []
            JOINT2D_CPN_AFFINED_NORMED = []
            JOINT3D_IMAGE_AFFINED = []
            for skeleton_indices_path in skeletons:
                """
                if isinstance(skeleton, str):
                    data = np.load(skeleton)
                elif isinstance(skeleton, np.ndarray):
                    data = skeleton
                else:
                    raise ValueError("Unsupported skeleton input type.")

                data = torch.from_numpy(data).float().unsqueeze(0)
                encoded_skeleton_indices, quant_t_shape = vqvae_encoder.encode(data)

                # 3. 从返回的形状中提取 t, h, w
                #    根据 vqvae.py, 输入是 [B, C, T, J], T是时间/帧, J是关节
                #    所以 t=1 (单个序列), h=T_quant, w=J_quant
                #    quant_t_shape is [B, C, T_quant, J_quant]
                _, _, t_quant, j_quant = quant_t_shape
                grid_shape = [1, t_quant, j_quant] # [t, h, w]
                all_grid_shapes.append(grid_shape)
                # results.append(encoded_skeleton_indices.squeeze(0)) # 先去掉batch_size维度再将时空空间压缩成一维
                results.append(encoded_skeleton_indices.squeeze(0))
                """
                assert 'skeleton_code' in skeleton_indices_path
                skeleton_indices = np.load(skeleton_indices_path)
                skeleton_indices = torch.from_numpy(skeleton_indices)
                CODEBOOK_INDICES.append(skeleton_indices)

                skeleton_pose3d_path = skeleton_indices_path.replace('skeleton_code', 'skeleton_pose3d')    # 与 generate_img2skel_data.py 中的保存子文件夹名称保持一致
                skeleton_pose3d = np.load(skeleton_pose3d_path)  # [T, 17, 3]
                skeleton_pose3d = torch.from_numpy(skeleton_pose3d)
                POSES.append(skeleton_pose3d)

                quant_shape_path = skeleton_indices_path.replace('skeleton_code', 'skeleton_quant_shape')
                quant_shape = np.load(quant_shape_path)  # [3], [C, T_quant, J_quant]
                _, t_quant, j_quant = quant_shape
                grid_shape = [1, t_quant, j_quant] # [t, h, w]
                GRID_SHAPES.append(grid_shape)

                source_slice_id_path = skeleton_indices_path.replace('skeleton_code', 'source_slice_id')
                if os.path.exists(source_slice_id_path):
                    source_slice_id = np.load(source_slice_id_path)  # [3], [C, T_quant, J_quant]
                    source_slice_id = torch.from_numpy(source_slice_id)
                    SOURCE_SLICE_ID.append(source_slice_id)

                joint3d_image_affined_path = skeleton_indices_path.replace('skeleton_code', 'joint3d_image_affined')
                if os.path.exists(joint3d_image_affined_path):
                    joint3d_image_affined = np.load(joint3d_image_affined_path)  # [3], [C, T_quant, J_quant]
                    joint3d_image_affined = torch.from_numpy(joint3d_image_affined)
                    JOINT3D_IMAGE_AFFINED.append(joint3d_image_affined)

                joint2d_cpn_path = skeleton_indices_path.replace('skeleton_code', 'joint2d_cpn')
                if os.path.exists(joint2d_cpn_path):
                    joint2d_cpn = np.load(joint2d_cpn_path)  # [3], [C, T_quant, J_quant]
                    joint2d_cpn = torch.from_numpy(joint2d_cpn)

                norm_offset_path = skeleton_indices_path.replace('skeleton_code', 'norm_transl')
                if os.path.exists(norm_offset_path):
                    norm_offset = np.load(norm_offset_path)
                    norm_offset = torch.from_numpy(norm_offset)

                norm_scale_path = skeleton_indices_path.replace('skeleton_code', 'norm_scale')
                if os.path.exists(norm_scale_path):
                    norm_scale = np.load(norm_scale_path)
                    norm_scale = torch.from_numpy(norm_scale)

                affine_trans_path = skeleton_indices_path.replace('skeleton_code', 'affine_trans')
                if os.path.exists(affine_trans_path):
                    affine_trans = np.load(affine_trans_path)
                    affine_trans = torch.from_numpy(affine_trans)

                if os.path.exists(joint2d_cpn_path) and os.path.exists(norm_offset_path) and os.path.exists(norm_scale_path) and os.path.exists(affine_trans_path):
                    joint2d_cpn_xy1 = torch.cat([joint2d_cpn, joint2d_cpn.new_ones(joint2d_cpn[..., :1].shape)], dim=-1)    # [T,17,3]
                    joint2d_cpn_affined = torch.einsum('tij,tkj->tik', joint2d_cpn_xy1, affine_trans)
                    joint2d_cpn_affined_normed = joint2d_cpn_affined / norm_scale[..., None, :2] - norm_offset[..., None, :2]
                    JOINT2D_CPN_AFFINED_NORMED.append(joint2d_cpn_affined_normed)
                    
                    

            return {
                "skeleton_indices": CODEBOOK_INDICES,  # 相当于 image 模态对应的 pixel_values
                "skeleton_poses": POSES,
                "skeleton_grid_thw": torch.tensor(GRID_SHAPES, dtype=torch.long),
                "source_slice_id": SOURCE_SLICE_ID,
                "joint2d_cpn_affined_normed": JOINT2D_CPN_AFFINED_NORMED,   # 记得改模型forward函数
                "joint3d_image_affined": JOINT3D_IMAGE_AFFINED,   # 记得改模型forward函数
                }
    
        else:
            SKEL_DICT = defaultdict(list)
            for skel_item in skeletons:
                data_key = skel_item['data_key']
                data_aux_keys = skel_item['data_aux_key']
                st_id = skel_item['st_id']
                ed_id = skel_item['ed_id']
                sample_id = skel_item['sample_id']

                sample_dict = self.mocap_dataset[sample_id]
                SKEL_DICT[data_key].append(sample_dict[data_key])
                for data_aux_key in data_aux_keys:
                    if data_aux_key in sample_dict.keys():
                        SKEL_DICT[data_aux_key].append(sample_dict[data_aux_key])

                skeleton_indices = self.vqvae_output[f"{data_key}_code"][sample_id]
                quant_shape = self.vqvae_output['quant_shape'][sample_id]
                _, t_quant, j_quant = quant_shape
                grid_shape = np.array([1, t_quant, j_quant]) # [t, h, w]
                SKEL_DICT['skeleton_indices'].append(skeleton_indices)
                SKEL_DICT['skeleton_grid_thw'].append(grid_shape)
            
            for key in SKEL_DICT:
                try:
                    SKEL_DICT[key] = np.stack(SKEL_DICT[key], axis=0)
                    SKEL_DICT[key] = torch.from_numpy(SKEL_DICT[key])
                except:
                    pass
            
            return SKEL_DICT
                
                

    
    #########################################################################################


    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        skeletons: list["SkeletalInput"], # ADDED BY BRADLEY 250827
        processor: "MMProcessor",
        imglens: Optional[list[int]] = None,
    ) -> dict[str, "torch.Tensor"]:
        r"""Process visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height
                            where num_patches == torch.prod(image_grid_thw)

        Returns: (mllama)
            pixel_values: tensor with shape
                          (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                          For example, (2, 1, 4, 3, 560, 560).
            aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
            aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
            num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).

        """
        mm_inputs = {}
        if len(images) != 0:
            image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if imglens is not None:  # if imglens are provided, make batched images
                images = _make_batched_images(images, imglens)

            image_processor_kwargs = {}
            if getattr(processor, "image_do_pan_and_scan", False):  # gemma3 image processor
                image_processor_kwargs.update(
                    {
                        "do_pan_and_scan": True,
                        "pan_and_scan_min_crop_size": 256,
                        "pan_and_scan_max_num_crops": 4,
                        "pan_and_scan_min_ratio_to_activate": 1.2,
                    }
                )

            mm_inputs.update(image_processor(images, return_tensors="pt", **image_processor_kwargs))

        if len(videos) != 0:
            video_processor: BaseImageProcessor = getattr(
                processor, "video_processor", getattr(processor, "image_processor", None)
            )
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            if "videos" in inspect.signature(video_processor.preprocess).parameters:  # for qwen2_vl and video_llava
                mm_inputs.update(video_processor(images=None, videos=videos, return_tensors="pt"))
            else:  # for llava_next_video
                mm_inputs.update(video_processor(videos, return_tensors="pt"))

        if len(audios) != 0:
            feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask", None)  # prevent conflicts
        
        # ADDED BY BRADLEY 250827 ###############################################################
        if len(skeletons) != 0: # New skeleton processing
            raise NotImplementedError
            skeleton_data = self._regularize_skeletons(skeletons, processor)

            # 1. 从返回的字典中提取 grid_shapes
            skeleton_grid_shapes = skeleton_data.pop("skeleton_grid_shapes")
            
            # 2. 将形状列表转换为 Qwen-VL 需要的 Tensor 格式
            #    每个 shape 应该是 [t, h, w]
            mm_inputs["skeleton_grid_thw"] = torch.tensor(skeleton_grid_shapes, dtype=torch.long)

            mm_inputs.update(skeleton_data)
        #########################################################################################

        return mm_inputs


@dataclass
class BasePlugin(MMPluginMixin):
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        skeletons: list["SkeletalInput"], # ADDED BY BRADLEY 250827
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        r"""Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios, skeletons)  # MODIFIED BY BRADLEY 250827
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        skeletons: list["SkeletalInput"], # ADDED BY BRADLEY 250827
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], Optional[list[int]]]:
        r"""Pre-process token ids after tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios, skeletons)  # MODIFIED BY BRADLEY 250827
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        skeletons: list["SkeletalInput"], # ADDED BY BRADLEY 250827
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        skel_lens: list[int], # ADDED BY BRADLEY 250827
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        r"""Build batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            audios: a list of audio inputs, shape (num_audios,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            audlens: number of audios in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos

        """
        self._validate_input(processor, images, videos, audios, skeletons)  # MODIFIED BY BRADLEY 250827
        return self._get_mm_inputs(images, videos, audios, skeletons, processor)    # MODIFIED BY BRADLEY 250827


@dataclass
class Qwen2VLPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height))

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height))

        return image

    @override
    def _regularize_videos(
        self, videos: list["VideoInput"], **kwargs
    ) -> dict[str, Union[list[list["ImageObject"]], list[float]]]:
        results, fps_per_video = [], []
        for video in videos:
            frames: list[ImageObject] = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")

                frames = video
                fps_per_video.append(kwargs.get("video_fps", 2.0))
            else:
                container = av.open(video, "r")
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())

                if video_stream.duration is None:
                    fps_per_video.append(kwargs.get("video_fps", 2.0))
                else:
                    fps_per_video.append(len(sample_indices) / float(video_stream.duration * video_stream.time_base))

            if len(frames) % 2 != 0:
                frames.append(frames[-1])

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {"videos": results, "fps_per_video": fps_per_video}

    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        skeletons: list["SkeletalInput"], # ADDED BY BRADLEY 250827
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)   # <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl_fast.Qwen2VLImageProcessorFast'>
        # from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            mm_inputs.update(image_processor(images=None, videos=video_data["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in video_data["fps_per_video"]]

        # ADDED BY BRADLEY 251002 ###############################################################
        if not hasattr(self, 'mocap_dataset'):
            dataset_file = skeletons[0]['dataset_file']
            dataset_args_file = dataset_file.replace('_data.jsonl', '_dataset_args.json')
            with open(dataset_args_file, 'r') as f:
                dataset_args = json.load(f)
            
            print("\nLoading dataset...", end=' ')
            dataset_loading_time_st = time()
            mocap_dataset = Multimodal_Mocap_Dataset(
                **dataset_args
            )
            print(f"Took {time()-dataset_loading_time_st:.1f} seconds\n")
            setattr(self, 'mocap_dataset', mocap_dataset)

            vqvae_output_file = dataset_file.replace('_data.jsonl', '_vqvae_output.pkl')
            vqvae_output = joblib.load(vqvae_output_file)
            setattr(self, 'vqvae_output', vqvae_output)

            prompt_config_file = skeletons[0]['dataset_file'].replace('_data.jsonl', '_prompt_config.json')
            with open(prompt_config_file, 'r') as f:
                prompt_config = json.load(f)
            task_name = prompt_config['task']
            setattr(self, 'task_name', task_name)
            prompt_type = prompt_config['prompt_type']
            setattr(self, 'prompt_type', prompt_type)
            get_skel_str_func = prompt_config['get_skel_str_func']
            setattr(self, 'get_skel_str_func', get_skel_str_func)


            vqvae_data_key = skeletons[0]['data_key']
            setattr(self, 'vqvae_data_key', vqvae_data_key)

        # task_pattern = re.compile(r"<\|task:(.*?)\|>")
        # task_names = task_pattern.findall(messages[0]['content'])
        # assert len(task_names) == 1
        # task_name = task_names[0]
        # messages[0]['content'] = messages[0]['content'].replace(f"<|task:{task_name}|>", "")

        # prompt_type_pattern = re.compile(r"<\|prompt_type:(.*?)\|>")
        # prompt_types = prompt_type_pattern.findall(messages[0]['content'])
        # assert len(prompt_types) == 1
        # prompt_type = prompt_types[0]
        # messages[0]['content'] = messages[0]['content'].replace(f"<|prompt_type:{prompt_type}|>", "")

        #########################################################################################


        # ADDED BY BRADLEY 250827 ###############################################################
        if len(skeletons) != 0: # New skeleton processing
            skeleton_data = self._regularize_skeletons(skeletons)
            mm_inputs.update(skeleton_data)
        #########################################################################################

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        skeletons: list["SkeletalInput"], # ADDED BY BRADLEY 250827
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        # self._validate_input(processor, images, videos, audios, skeletons)  # MODIFIED BY BRADLEY 250827
        # self._validate_messages(messages, images, videos, audios, skeletons)  # MODIFIED BY BRADLEY 250827
        num_image_tokens, num_video_tokens = 0, 0
        num_skeleton_tokens = 0 # ADDED BY BRADLEY 250827
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, skeletons, processor)   # MODIFIED BY BRADLEY 250827
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            skeleton_grid_thw = mm_inputs.get("skeleton_grid_thw", [])  # ADDED BY BRADLEY 250828
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            skeleton_grid_thw = [None] * len(skeletons)  # ADDED BY BRADLEY 250828

        for message in messages:
            content = message["content"]

            # ADDED BY BRADLEY 251002 ###############################################################
            while PROMPT_PLACEHOLDER in content:
                prompt_templates = PROMPT_TEMPLATES[self.task_name][self.prompt_type]
                prompt_template = prompt_templates[0]
                content = content.replace(PROMPT_PLACEHOLDER, prompt_template, 1)
            while RESPONSE_PLACEHOLDER in content:
                raise NotImplementedError
            #########################################################################################

            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER, f"<|vision_start|>{self.video_token * video_seqlen}<|vision_end|>", 1
                )
                num_video_tokens += 1

            # ADDED BY BRADLEY 250827 ###############################################################
            while SKELETON_PLACEHOLDER in content: # New
                """
                # skeleton_seqlen = skeleton_grid_thw[num_skeleton_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                # For skeletons, we do not merge, so we remove `// merge_length`. 需要与 get_rope_indedx (extras_byBrad/get_rope_index.py) 中的 spatial merge 逻辑保持一致
                skeleton_seqlen = skeleton_grid_thw[num_skeleton_tokens].prod() if self.expand_mm_tokens else 1
                content = content.replace(
                    SKELETON_PLACEHOLDER, f"<|skel_start|>{self.skeleton_token * skeleton_seqlen}<|skel_end|>", 1
                )
                num_skeleton_tokens += 1
                """

                # 以下是另一种思路的实现
                if self.expand_mm_tokens:

                    if isinstance(skeletons[0], str) and skeletons[0].endswith('.npy') and 'skeleton_code' in skeletons[0]:
                        # Get the skeleton indices and convert to a token string
                        skeleton_indices = mm_inputs["skeleton_indices"][num_skeleton_tokens]    # list of (T,J). e.g., (4,17)
                        joint3d_image_affined = mm_inputs["joint3d_image_affined"][num_skeleton_tokens]
                        
                        get_skel_str_func = globals()[processor.skeleton_processor]

                        if 'coord' in processor.skeleton_processor:
                            skeleton_input = joint3d_image_affined
                        else:
                            skeleton_input = skeleton_indices

                        skeleton_token_str = get_skel_str_func(skeleton_input)  # e.g., "<skel_0><skel_1>...<skel_16><|><skel_0>..."

                        content = content.replace(
                            SKELETON_PLACEHOLDER, f"<|skel_start|>{skeleton_token_str}<|skel_end|>", 1,
                        )

                    else:
                        # Get the skeleton indices and convert to a token string
                        skeleton_input = mm_inputs[self.get_skel_str_func['input']][num_skeleton_tokens]    # list of (T,J). e.g., (4,17)

                        get_skel_str_func = getattr(convert_skel_token_v2, self.get_skel_str_func['name'])

                        extra_inputs = dict(task=self.task_name)
                        if self.task_name == 'SkelPred':
                            assert num_skeleton_tokens == 0 or num_skeleton_tokens == 1
                            extra_inputs['task_context'] = 'history' if num_skeleton_tokens == 0 else 'future'
                        skeleton_token_str = get_skel_str_func(skeleton_input, **extra_inputs)  # e.g., "<skel_0><skel_1>...<skel_16><|><skel_0>..."

                        content = content.replace(
                            SKELETON_PLACEHOLDER, f"<|skel_start|>{skeleton_token_str}<|skel_end|>", 1,
                        )
                        

                else:
                    raise NotImplementedError("Non-expanded skeleton tokens not implemented yet.")
                    self.skeleton_token = '<|pad|>'
                    content = content.replace(SKELETON_PLACEHOLDER, self.skeleton_token, 1)

                num_skeleton_tokens += 1
            #########################################################################################

            message["content"] = content

        return messages

PLUGINS = {
    "base": BasePlugin,
    "qwen2_vl": Qwen2VLPlugin,
}


def register_mm_plugin(name: str, plugin_class: type["BasePlugin"]) -> None:
    r"""Register a multimodal plugin."""
    if name in PLUGINS:
        raise ValueError(f"Multimodal plugin {name} already exists.")

    PLUGINS[name] = plugin_class


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
    audio_token: Optional[str] = None,
    skeleton_token: Optional[str] = None, # ADDED BY BRADLEY 250827
) -> "BasePlugin":
    r"""Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token, skeleton_token) # MODIFIED BY BRADLEY 250827
