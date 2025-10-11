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


# import debugpy
# debugpy.listen(('0.0.0.0', 5678))
# debugpy.wait_for_client()

import json
import joblib
from time import time
from collections import defaultdict
from typing import TYPE_CHECKING, Optional
import re
import numpy as np
import torch
import os
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

import _llamafactory_skeleton_byBrad.data_utils.convert_skel_token as convert_skel_token_v2
from ...extras_byBrad.convert_skel_token import *
import sys
sys.path.append('../Skeleton-in-Context-tpami/')
from lib.utils.viz_skel_seq import viz_skel_seq_anim # type: ignore
sys.path.remove('../Skeleton-in-Context-tpami/')
sys.path.append("../ContextAwarePoseFormer_Private/H36M-Toolbox/")
from multimodal_h36m_dataset_byBradley import Multimodal_Mocap_Dataset, DATA_ROOT_PATH
sys.path.remove("../ContextAwarePoseFormer_Private/H36M-Toolbox/")
    


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):


    print('\npython ' + ' '.join(sys.argv))
    print('\nPID: ' + str(os.getpid()))


    if data_args.dataset_eval_range is not None:
        dataset_start_id, dataset_end_id = data_args.dataset_eval_range
        print('\n\n--------------------------------------------------------------')
        print(f'Testing sub-dataset {dataset_start_id} -- {dataset_end_id}')
        print('--------------------------------------------------------------\n\n')




    if 'debugpy' in sys.modules:
        data_args.max_samples = 512
        # training_args.per_device_train_batch_size = 1
        data_args.preprocessing_num_workers = 1
        # training_args.per_device_eval_batch_size = 1
        # training_args.per_device_train_batch_size = 1
        training_args.dataloader_num_workers = 1
        # training_args.save_steps=1
        pass
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]   # <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    



    try:
        dataset_file = data_args.dataset_dir['placeholder']['file_name']
        assumed_data_split = 'test' if 'test' in dataset_file else 'train'
        dataset_dir = os.path.dirname(dataset_file)
        dataset_args_file = os.path.join(dataset_dir, assumed_data_split+'_dataset_args.json')
        with open(dataset_args_file, 'r') as f:
            dataset_args = json.load(f)
        print("\nLoading dataset...", end=' ')
        dataset_loading_time_st = time()
        if 'video_rgb' in dataset_args['get_item_list']:
            dataset_args['get_item_list'] = [get_item for get_item in dataset_args['get_item_list'] if get_item != 'video_rgb']
        mocap_dataset = Multimodal_Mocap_Dataset(
            **dataset_args
        )
        print(f"Took {time()-dataset_loading_time_st:.1f} seconds\n")
        vqvae_output_file = os.path.join(dataset_dir, assumed_data_split+'_vqvae_output.pkl')
        if os.path.exists(vqvae_output_file):
            vqvae_output = joblib.load(vqvae_output_file)

        prompt_config_file = os.path.join(dataset_dir, assumed_data_split+'_prompt_config.json')
        with open(prompt_config_file, 'r') as f:
            prompt_config = json.load(f)
        task_name = prompt_config['task']
        prompt_type = prompt_config['prompt_type']
        get_skel_str_func_info = prompt_config['get_skel_str_func']


        setattr(template.mm_plugin, 'mocap_dataset', mocap_dataset)
        if os.path.exists(vqvae_output_file):
            setattr(template.mm_plugin, 'vqvae_output', vqvae_output)
        setattr(template.mm_plugin, 'task_name', task_name)
        setattr(template.mm_plugin, 'prompt_type', prompt_type)
        setattr(template.mm_plugin, 'get_skel_str_func', get_skel_str_func_info)

    except Exception as e:
        print(f'Error loading mocap dataset or vqvae output or prompt config: {e}')
        pass


           
    
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

        if get_skel_str_func_info['input'] == 'skeleton_indices':
            if model_args.vqvae_ckpt is not None:
                print('\n'.join(['Warning!!! using vqvae from llamafactory.extras_byBrad.vqvae is deprecated, please use `hybrid_vqvae` instead.' for _ in range(99)]))

                from llamafactory.extras_byBrad.vqvae import SKEL_VQVAE as SkeletonProcessor, Encoder, VectorQuantizer, Decoder
                from safetensors.torch import load_file
                encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1])
                vq = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)
                decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
                skeleton_processor = SkeletonProcessor(encoder, decoder, vq)

                ckpt_path = model_args.vqvae_ckpt
                state_dict = load_file(ckpt_path, device="cpu")
                skeleton_processor.load_state_dict(state_dict)
                skeleton_processor.eval()
                for param in skeleton_processor.parameters():
                    param.requires_grad = False
                skeleton_processor = skeleton_processor.cuda()
            elif model_args.vqvae_config is not None:
                from safetensors.torch import load_file as load_safetensors
                sys.path.append('..//MTVCrafter/')
                from models import HYBRID_VQVAE # type: ignore
                sys.path.remove('..//MTVCrafter/')
                skeleton_processor = HYBRID_VQVAE(model_args.vqvae_config.vqvae_config.encoder,
                                                model_args.vqvae_config.vqvae_config.decoder,
                                                model_args.vqvae_config.vqvae_config.vq, 
                                                vision_config=model_args.vqvae_config.vision_config, 
                                                joint_data_type=model_args.vqvae_config.vqvae_config.joint_data_type,
                                                )
                vqvae_ckpt = model_args.vqvae_config.vqvae_config.resume_path
                skeleton_processor.load_model_weights(vqvae_ckpt)
                skeleton_processor.eval()
                for param in skeleton_processor.parameters():
                    param.requires_grad = False
                skeleton_processor = skeleton_processor.cuda()
            elif os.path.exists(os.path.join(os.path.dirname(sys.argv[1]), 'vqvae_config.py')):
                try:
                    vqvae_config_python_file = os.path.join(os.path.dirname(sys.argv[1]), 'vqvae_config.py')
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("vqvae_config_module", vqvae_config_python_file)
                    vqvae_config_module = importlib.util.module_from_spec(spec)
                    sys.modules["vqvae_config_module"] = vqvae_config_module
                    spec.loader.exec_module(vqvae_config_module)
                    from easydict import EasyDict as edict
                    setattr(model_args, 'vqvae_config', edict(
                        vqvae_config=vqvae_config_module.vqvae_config,
                        vision_config=vqvae_config_module.vision_config,
                    ))

                    from safetensors.torch import load_file as load_safetensors
                    sys.path.append('..//MTVCrafter/')
                    from models import HYBRID_VQVAE # type: ignore
                    sys.path.remove('..//MTVCrafter/')
                    skeleton_processor = HYBRID_VQVAE(model_args.vqvae_config.vqvae_config.encoder,
                                                    model_args.vqvae_config.vqvae_config.decoder,
                                                    model_args.vqvae_config.vqvae_config.vq, 
                                                    vision_config=model_args.vqvae_config.vision_config, 
                                                    joint_data_type=model_args.vqvae_config.vqvae_config.joint_data_type,
                                                    )
                    vqvae_ckpt = model_args.vqvae_config.vqvae_config.resume_path
                    skeleton_processor.load_model_weights(vqvae_ckpt)
                    skeleton_processor.eval()
                    for param in skeleton_processor.parameters():
                        param.requires_grad = False
                    skeleton_processor = skeleton_processor.cuda()
                except Exception as e:
                    print(e)
            else:
                raise NotImplementedError
        

        """
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
            sys.path.append('/group/40174/peimingli/bradley//Skeleton-in-Context-tpami/')
            from lib.utils.viz_skel_seq import viz_skel_seq_anim

            # viz_skel_seq_anim({'pred': custom_predict_motion}, fs=0.5, if_print=False)
            # viz_skel_seq_anim({'pred': custom_predict_motion}, fs=0.5, if_print=True, fig_title=f"custom_sample", file_folder='.', file_name=f'custom_sample')
        except:
            print(f'[Custom Sample] shape mismatch: {[len(tmp) for tmp in custom_predict_motion_id]}. Skipping this sample')
        ################################################################################################################################
        """

        # get_skel_str_func = globals()[model_args.get_skel_str_func]
        # same as: globals()[tokenizer_module['processor']['skeleton_processor']]
        # parse_skel_str_func = globals()[model_args.parse_skel_str_func]

        get_skel_str_func = getattr(convert_skel_token_v2, get_skel_str_func_info['name'])
        parse_skel_str_func = getattr(convert_skel_token_v2, get_skel_str_func_info['name'].replace('get_', 'parse_'))


        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")

        if data_args.dataset_eval_range is not None:
            from datasets import Dataset
            dataset_start_id, dataset_end_id = data_args.dataset_eval_range
            dataset_module["eval_dataset"] = Dataset.from_dict(dataset_module["eval_dataset"][dataset_start_id:dataset_end_id])
        else:
            dataset_start_id, dataset_end_id = 'all', 'all'


        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        """
        from datasets import Dataset
        pred_results = trainer.predict(Dataset.from_dict(dataset_module["eval_dataset"][0:1]), metric_key_prefix="predict", **gen_kwargs)
        """
        # print(tokenizer.decode(dataset_module["eval_dataset"][0]['input_ids']).replace('<|video_pad|>',''))
        # predict_results[0]: predictions. (10, 1133)
        # predict_results[1]: label_ids. (10, 96)
        MOTION_LABEL = []
        MOTION_PRED = []
        success_log = []
        for sample_id in range(predict_results.predictions.shape[0]):
            sample_prediction = predict_results.predictions[sample_id]  # (1133,)
            sample_label = predict_results.label_ids[sample_id] # (96,)

            text_prediction = tokenizer.decode(sample_prediction[sample_prediction != -100], skip_special_tokens=False)
            text_prediction = text_prediction.replace('<|endoftext|>', '')
            text_label = tokenizer.decode(sample_label[sample_label != -100], skip_special_tokens=False)


            motion_id_label = parse_skel_str_func(text_label)
            motion_id_label = np.array(motion_id_label)
            motion_id_label = torch.from_numpy(motion_id_label).long().unsqueeze(0).cuda()  # (1, quan_t, 17)
            if get_skel_str_func_info['input'] == 'skeleton_indices':
                motion_label = skeleton_processor.decode(motion_id_label).squeeze(0).cpu().numpy()  # (T, 17, 3)
            else:
                motion_label = motion_id_label.cpu().numpy()


            try:
                motion_id_prediction = parse_skel_str_func(text_prediction)
                motion_id_prediction = np.array(motion_id_prediction)
                motion_id_prediction = torch.from_numpy(motion_id_prediction).long().unsqueeze(0).cuda()  # (1, quan_t, 17)
                if get_skel_str_func_info['input'] == 'skeleton_indices':
                    motion_prediction = skeleton_processor.decode(motion_id_prediction).squeeze(0).cpu().numpy()  # (T, 17, 3)
                else:
                    motion_prediction = motion_id_prediction.cpu().numpy()

                MOTION_LABEL.append(motion_label)
                MOTION_PRED.append(motion_prediction)
                success_log.append(sample_id)



                # source_slice_id_path = dataset_module["eval_dataset"][sample_id]['skeletons'][0].replace('skeleton_code', 'source_slice_id')
                # if os.path.exists(source_slice_id_path):
                #     source_slice_id = np.load(source_slice_id_path)   # (T,)

                #     if 'h36m_data' not in locals():
                #         import joblib
                #         h36m_data = joblib.load("/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl")
                    
                #     camera_name = h36m_data['test']['camera_name'][source_slice_id]



                # viz_skel_seq_anim({'gt': motion_label, 'pred': motion_prediction}, fs=0.5, if_print=False)
                # viz_skel_seq_anim({'gt': motion_label, 'pred': motion_prediction}, fs=0.5, if_print=True, fig_title=f"{sample_id}", file_folder='.', file_name=f'{sample_id:06d}')

                
            except Exception as e:
                print(f'[SampleID {sample_id}] {e}. Skipping this sample')
                continue

        MOTION_LABEL=np.stack(MOTION_LABEL,axis=0)        # [N,T,17,3]
        try: 
            MOTION_PRED=np.stack(MOTION_PRED,axis=0)          # [N,T,17,3]
        except ValueError as e:
            valid_shape_indices = []
            new_motion_pred = []
            for pred_id, motion_pred_ in enumerate(MOTION_PRED):
                if motion_pred_.shape == MOTION_LABEL[0].shape:
                    valid_shape_indices.append(pred_id)
                    new_motion_pred.append(motion_pred_)
            MOTION_PRED = np.stack(new_motion_pred,axis=0)
            MOTION_LABEL = MOTION_LABEL[valid_shape_indices]

            success_log = np.array(success_log)[valid_shape_indices].tolist()

        mpjpe_all = np.linalg.norm((MOTION_LABEL - MOTION_LABEL[...,0:1,:])
                                   - (MOTION_PRED - MOTION_PRED[...,0:1,:]), axis=-1).mean((-2,-1)) # (N,)
        mpjpe_all = mpjpe_all * 1000
        print(f'avg mpjpe_all: ({mpjpe_all.shape} samples)', mpjpe_all.mean())


        try:
            if isinstance(dataset_module["eval_dataset"]['skeletons'][0][0], str):
                skeleton_npy_path_list = sum(dataset_module["eval_dataset"]['skeletons'],[])
                skeleton_scale = np.stack([np.load(skeleton_npy_path.replace('skeleton_code','norm_scale')) for skeleton_npy_path in skeleton_npy_path_list])[..., None, :]     # (N,T,1,3)
                skeleton_offset = np.stack([np.load(skeleton_npy_path.replace('skeleton_code','norm_transl')) for skeleton_npy_path in skeleton_npy_path_list])[..., None, :]     # (N,T,1,3)
                MOTION_GT_PIX_NORMED = np.stack([np.load(skeleton_npy_path.replace('skeleton_code','skeleton_pose3d')) for skeleton_npy_path in skeleton_npy_path_list])    # (N,T,1,3)
                trans_inv = np.stack([np.load(skeleton_npy_path.replace('skeleton_code','affine_trans_inv')) for skeleton_npy_path in skeleton_npy_path_list])  # (N,T,2,3)
                factor_2_5d = np.stack([np.load(skeleton_npy_path.replace('skeleton_code','factor_2_5d')) for skeleton_npy_path in skeleton_npy_path_list])[..., None, None]     # (N,T,1,1)
            else:
                if task_name == 'SkelPred':
                    history_skel_info_dict_list = [dataset_module["eval_dataset"]['skeletons'][local_sample_id][0] for local_sample_id in range(len(dataset_module["eval_dataset"]['skeletons']))]
                    skel_info_dict_list = [dataset_module["eval_dataset"]['skeletons'][local_sample_id][1] for local_sample_id in range(len(dataset_module["eval_dataset"]['skeletons']))]
                elif task_name == 'Vid2Skel-SkelPred-TwoTurn':
                    history_skel_info_dict_list = [dataset_module["eval_dataset"]['skeletons'][local_sample_id][0] for local_sample_id in range(len(dataset_module["eval_dataset"]['skeletons']))]
                    if len(dataset_module["eval_dataset"]['skeletons'][0]) == 2:
                        skel_info_dict_list = [dataset_module["eval_dataset"]['skeletons'][local_sample_id][1] for local_sample_id in range(len(dataset_module["eval_dataset"]['skeletons']))]
                    else:
                        print('!!!!!! It seems like you are testing >>>Vid2Skel<<< task inside a >>>Vid2Skel-SkelPred-TwoTurn<<< project. Check code if it is not what you intend\n'*50)
                        skel_info_dict_list = history_skel_info_dict_list
                else:
                    skel_info_dict_list = sum(dataset_module["eval_dataset"]['skeletons'],[])
                SKEL_DICT = defaultdict(list)

                # for idx in range(len(skel_info_dict_list)):
                for idx in success_log:
                    skel_item = skel_info_dict_list[idx]

                    data_key = skel_item['data_key']
                    data_aux_keys = skel_item['data_aux_key']
                    st_id = skel_item['st_id']
                    ed_id = skel_item['ed_id']
                    sample_id = skel_item['sample_id']

                    sample_dict = mocap_dataset[sample_id]
                    SKEL_DICT[data_key].append(sample_dict[data_key])
                    for data_aux_key in data_aux_keys:
                        if data_aux_key == data_key:
                            continue
                        if data_aux_key in sample_dict.keys():
                            SKEL_DICT[data_aux_key].append(sample_dict[data_aux_key])
                    # skeleton_indices = vqvae_output[f"{data_key}_code"][sample_id]
                    # history_skeleton_indices = vqvae_output[f"{data_key}_code"][history_skel_info_dict_list[idx]['sample_id']]

                for key in SKEL_DICT:
                    try:
                        SKEL_DICT[key] = np.stack(SKEL_DICT[key], axis=0)
                    except:
                        pass
                

                MOTION_GT_PIX_NORMED = SKEL_DICT[data_key]
                if 'image' in data_key:
                    skeleton_scale = SKEL_DICT[data_key.replace('_normed', '_scale')][..., None, :]
                    skeleton_offset = SKEL_DICT[data_key.replace('_normed', '_transl')][..., None, :]
                    trans_inv = SKEL_DICT['affine_trans_inv']
                    factor_2_5d = SKEL_DICT['factor_2_5d'][..., None, None]
                


            if 'joint3d_cam' in data_key:
                MOTION_PRED_MM = MOTION_PRED * 1000
                MOTION_LABEL_MM = MOTION_LABEL * 1000
                MOTION_GT_MM = MOTION_GT_PIX_NORMED * 1000
                
            elif 'image' in data_key:
                MOTION_PRED_PIX_AFFINED = (MOTION_PRED + skeleton_offset) * skeleton_scale
                MOTION_LABEL_PIX_AFFINED = (MOTION_LABEL + skeleton_offset) * skeleton_scale
                MOTION_GT_PIX_AFFINED = (MOTION_GT_PIX_NORMED + skeleton_offset) * skeleton_scale

                mpjpe_all_pix_affined_avg = np.linalg.norm(MOTION_PRED_PIX_AFFINED - MOTION_LABEL_PIX_AFFINED, axis=-1).mean()
                try:
                    save_path = os.path.dirname(sys.argv[1])
                    save_path = os.path.join(save_path, f'save_data_{dataset_start_id}-{dataset_end_id}')
                    os.makedirs(save_path, exist_ok=True)
                    np.save(os.path.join(save_path, f'pix_affined_pred_{mpjpe_all_pix_affined_avg:.1f}.npy'), MOTION_PRED_PIX_AFFINED)
                    np.save(os.path.join(save_path, f'pix_affined_label_{mpjpe_all_pix_affined_avg:.1f}.npy'), MOTION_LABEL_PIX_AFFINED)
                except Exception:
                    pass


                MOTION_PRED_PIX_AFFINED_xy1 = np.concatenate([MOTION_PRED_PIX_AFFINED[..., :2], np.ones_like(MOTION_PRED_PIX_AFFINED[..., :1])], axis=-1)
                MOTION_LABEL_PIX_AFFINED_xy1 = np.concatenate([MOTION_LABEL_PIX_AFFINED[..., :2], np.ones_like(MOTION_LABEL_PIX_AFFINED[..., :1])], axis=-1)
                MOTION_GT_PIX_AFFINED_xy1 = np.concatenate([MOTION_GT_PIX_AFFINED[..., :2], np.ones_like(MOTION_GT_PIX_AFFINED[..., :1])], axis=-1)

                MOTION_PRED_PIX_xy = np.einsum('btij,btkj->btki', trans_inv, MOTION_PRED_PIX_AFFINED_xy1)
                MOTION_LABEL_PIX_xy = np.einsum('btij,btkj->btki', trans_inv, MOTION_LABEL_PIX_AFFINED_xy1)
                MOTION_GT_PIX_xy = np.einsum('btij,btkj->btki', trans_inv, MOTION_GT_PIX_AFFINED_xy1)

                MOTION_PRED_PIX = np.concatenate([MOTION_PRED_PIX_xy, MOTION_PRED_PIX_AFFINED[..., 2:]], axis=-1)
                MOTION_LABEL_PIX = np.concatenate([MOTION_LABEL_PIX_xy, MOTION_LABEL_PIX_AFFINED[..., 2:]], axis=-1)
                MOTION_GT_PIX = np.concatenate([MOTION_GT_PIX_xy, MOTION_GT_PIX_AFFINED[..., 2:]], axis=-1)


                mpjpe_all_pix_avg = np.linalg.norm(MOTION_LABEL_PIX - MOTION_PRED_PIX, axis=-1).mean()
                try:
                    save_path = os.path.dirname(sys.argv[1])
                    save_path = os.path.join(save_path, f'save_data_{dataset_start_id}-{dataset_end_id}')
                    os.makedirs(save_path, exist_ok=True)
                    np.save(os.path.join(save_path, f'pix_pred_{mpjpe_all_pix_avg:.1f}.npy'), MOTION_PRED_PIX)
                    np.save(os.path.join(save_path, f'pix_label_{mpjpe_all_pix_avg:.1f}.npy'), MOTION_LABEL_PIX)
                except Exception:
                    pass

                MOTION_PRED_MM = MOTION_PRED_PIX * factor_2_5d
                MOTION_LABEL_MM = MOTION_LABEL_PIX * factor_2_5d
                MOTION_GT_MM = MOTION_GT_PIX * factor_2_5d

            

            MOTION_PRED_ROOTREL = MOTION_PRED_MM - MOTION_PRED_MM[...,0:1,:]
            MOTION_LABEL_ROOTREL = MOTION_LABEL_MM - MOTION_LABEL_MM[...,0:1,:]
            MOTION_GT_ROOTREL = MOTION_GT_MM - MOTION_GT_MM[...,0:1,:]
            mpjpe_all_mm = np.linalg.norm(MOTION_LABEL_ROOTREL - MOTION_PRED_ROOTREL, axis=-1).mean((-2,-1)) # (N,)
            mpjpe_all_mm_avg = mpjpe_all_mm.mean()
            try:
                save_path = os.path.dirname(sys.argv[1])
                save_path = os.path.join(save_path, f'save_data_{dataset_start_id}-{dataset_end_id}')
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, f'mm_pred_{mpjpe_all_mm_avg:.1f}.npy'), MOTION_PRED_ROOTREL)
                np.save(os.path.join(save_path, f'mm_label_{mpjpe_all_mm_avg:.1f}.npy'), MOTION_LABEL_ROOTREL)
            except Exception:
                pass

            if 'image' in data_key:
                print(f'{mpjpe_all_mm.shape} samples; avg mpjpe (pix; affined): ', mpjpe_all_pix_affined_avg)
                print(f'{mpjpe_all_mm.shape} samples; avg mpjpe (pix): ', mpjpe_all_pix_avg)

            print(f'{mpjpe_all_mm.shape} samples; avg mpjpe (mm): ', mpjpe_all_mm_avg)

        except:
            pass




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