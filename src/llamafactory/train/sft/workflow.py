# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional
import re
import numpy as np
import torch

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]   # <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)  # <class 'peft.peft_model.PeftModelForCausalLM'>
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    """Debug model and data_collator
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModel, Qwen2_5_VLTextModel
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from peft.tuners.lora.model import LoraModel
    from peft.peft_model import PeftModelForCausalLM
    _ = data_collator([dataset_module['train_dataset'][0], dataset_module['train_dataset'][1]])
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    """
    
    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:




        from llamafactory.extras_byBrad.vqvae import SKEL_VQVAE as SkeletonProcessor, Encoder, VectorQuantizer, Decoder
        from safetensors.torch import load_file
        encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1])
        vq = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)
        decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
        skeleton_processor = SkeletonProcessor(encoder, decoder, vq)
        mode = 'joint3d'
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


        

        '''
        # 这段代码是为了在一个独立的样本上进行推理, 不用数据集对象包装好的样本 ###############################################################
        new_video_frames_paths = [f"/data2/wxs/DATASETS/Human3.6M_MMPose/processed/images_fps50_cropped_192x256/S1/S1_Waiting_1.54138969/S1_Waiting_1.54138969_{img:06d}.jpg" 
                                  for img in range(1054,1070)]
        prompt_text = "Describe the following video <video> using text. Focus only on the actions and movements of the person."
        prompt_text = "Describe the following video <video> using skeleton tokens. Focus only on the actions and movements of the person."
        from PIL import Image        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        video_frames = [Image.open(p).convert("RGB") for p in new_video_frames_paths]
        messages[0]["content"].insert(1, {"type": "image", "content": video_frames})
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            custom_predict_ids = model.generate(**inputs, **gen_kwargs)
        input_token_len = inputs["input_ids"].shape[1]
        custom_predict_text = tokenizer.decode(custom_predict_ids[0, input_token_len:], skip_special_tokens=False)
        custom_predict_text = custom_predict_text.replace('<|endoftext|>', '')

        custom_predict_motion_id = extract_skeleton_tokens(custom_predict_text)
        try:
            custom_predict_motion_id = np.array(custom_predict_motion_id)
            custom_predict_motion_id = torch.from_numpy(custom_predict_motion_id).long().unsqueeze(0).cuda()  # (1, quan_t, 17)
            custom_predict_motion = skeleton_processor.decode(custom_predict_motion_id).squeeze(0).cpu().numpy()  # (T, 17, 3)

            import sys
            sys.path.append('/home/wxs/Skeleton-in-Context-tpami/')
            from lib.utils.viz_skel_seq import viz_skel_seq_anim

            # viz_skel_seq_anim({'pred': custom_predict_motion}, fs=0.5, if_print=False)
            # viz_skel_seq_anim({'pred': custom_predict_motion}, fs=0.5, if_print=True, fig_title=f"custom_sample", file_folder='.', file_name=f'custom_sample')
        except:
            print(f'[Custom Sample] shape mismatch: {[len(tmp) for tmp in custom_predict_motion_id]}. Skipping this sample')
        ################################################################################################################################
        '''




        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        # predict_results[0]: predictions. (10, 1133)
        # predict_results[1]: label_ids. (10, 96)

        MOTION_LABEL = []
        MOTION_PRED = []

        for sample_id in range(predict_results.predictions.shape[0]):
            sample_prediction = predict_results.predictions[sample_id]  # (1133,)
            sample_label = predict_results.label_ids[sample_id] # (96,)

            text_prediction = tokenizer.decode(sample_prediction[sample_prediction != -100], skip_special_tokens=False)
            text_prediction = text_prediction.replace('<|endoftext|>', '')
            text_label = tokenizer.decode(sample_label[sample_label != -100], skip_special_tokens=False)


            motion_id_prediction = extract_skeleton_tokens(text_prediction)
            motion_id_label = extract_skeleton_tokens(text_label)


            motion_id_label = np.array(motion_id_label)
            motion_id_label = torch.from_numpy(motion_id_label).long().unsqueeze(0).cuda()  # (1, quan_t, 17)
            motion_label = skeleton_processor.decode(motion_id_label).squeeze(0).cpu().numpy()  # (T, 17, 3)

            try:
                motion_id_prediction = np.array(motion_id_prediction)
                motion_id_prediction = torch.from_numpy(motion_id_prediction).long().unsqueeze(0).cuda()  # (1, quan_t, 17)
                motion_prediction = skeleton_processor.decode(motion_id_prediction).squeeze(0).cpu().numpy()  # (T, 17, 3)

                MOTION_LABEL.append(motion_label)
                MOTION_PRED.append(motion_prediction)
                continue

                import sys
                sys.path.append('/home/wxs/Skeleton-in-Context-tpami/')
                from lib.utils.viz_skel_seq import viz_skel_seq_anim

                viz_skel_seq_anim({'gt': motion_label, 'pred': motion_prediction}, fs=0.5, if_print=False)
                # viz_skel_seq_anim({'gt': motion_label, 'pred': motion_prediction}, fs=0.5, if_print=True, fig_title=f"{sample_id}", file_folder='.', file_name=f'{sample_id:06d}')

                
            except Exception as e:
                print(f'[SampleID {sample_id}] {e}. Skipping this sample')
                print(f'[SampleID {sample_id}] shape mismatch: {[len(tmp) for tmp in motion_id_prediction]}. Skipping this sample')
                continue


        MOTION_LABEL=np.stack(MOTION_LABEL,axis=0)        # [N,T,17,3]
        MOTION_PRED=np.stack(MOTION_PRED,axis=0)          # [N,T,17,3]

        mpjpe_all = np.linalg.norm((MOTION_LABEL - MOTION_LABEL[...,0:1,:])
                                   - (MOTION_PRED - MOTION_PRED[...,0:1,:]), axis=-1).mean((-2,-1)) # (N,)
        mpjpe_all = mpjpe_all * 1000




        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)



def extract_skeleton_tokens(text_label: str) -> list[list[int]]:
    """
    从包含结构化骨架token的字符串中严格提取数字索引。
    该函数只提取位于身体部位标签（如<torso>...</torso>）内部的骨架token，
    并根据<|frame_break|>进行分组。

    Args:
        text_label: 形如 '<|skel_start|><torso><skel_4318>...</torso>...<|frame_break|>...' 的字符串。

    Returns:
        一个嵌套列表，每个子列表包含一帧中的所有骨架token的数字索引。
        例如: [[4318, 7637, ...], [3553, 2218, ...], ...]
    """
    # 1. 定义用于匹配 <|skel_{i}|> 中数字的正则表达式
    skel_pattern = re.compile(r"<skel_(\d+)>")

    # 2. 定义用于匹配所有身体部位区块内容的正则表达式
    #    - <(torso|left_arm|right_arm|left_leg|right_leg)>: 匹配并捕获任一身体部位的起始标签
    #    - (.*?): 非贪婪地捕获起始和结束标签之间的所有内容
    #    - <\/\1>: 使用反向引用\1确保结束标签与起始标签匹配
    body_part_pattern = re.compile(r"<(torso|left_arm|right_arm|left_leg|right_leg)>(.*?)<\/\1>")

    # 3. (可选但更稳健) 先隔离出骨架数据的主体部分
    start_tag = "<|skel_start|>"
    end_tag = "<|skel_end|>"
    start_index = text_label.find(start_tag)
    end_index = text_label.find(end_tag)

    if start_index != -1 and end_index != -1:
        skeleton_block = text_label[start_index + len(start_tag) : end_index]
    else:
        skeleton_block = text_label

    # 4. 根据 <|frame_break|> 分割字符串，得到每一帧的字符串片段
    frame_strings = skeleton_block.split("<|frame_break|>")

    all_frames_indices = []
    # 5. 遍历每一帧的字符串片段
    for frame_str in frame_strings:
        # 6. 在当前帧中，找到所有身体部位区块及其内容
        #    findall会返回一个元组列表，例如 [('torso', '<skel_...><skel_...>'), ('left_arm', '<skel_...>')]
        part_contents = body_part_pattern.findall(frame_str)

        current_frame_indices = []
        # 7. 遍历找到的每个身体部位的内容
        for _, content in part_contents:  # 我们只需要内容，所以用 _ 忽略部位名称
            # 8. 在该部位的内容中提取所有骨架token的数字
            indices_as_strings = skel_pattern.findall(content)
            current_frame_indices.extend([int(index) for index in indices_as_strings])

        # 9. 如果该帧确实提取到了索引，则将其添加到最终结果中
        if current_frame_indices:
            all_frames_indices.append(current_frame_indices)

    return all_frames_indices