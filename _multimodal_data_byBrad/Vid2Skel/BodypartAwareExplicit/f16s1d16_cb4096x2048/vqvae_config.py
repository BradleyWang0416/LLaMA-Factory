from easydict import EasyDict as edict
import sys
sys.path.append('/home/wxs/MTVCrafter/')
from config.vision_backbone import config as vision_config # type: ignore
from config.vqvae import vqvae_config # type: ignore
# from models import HYBRID_VQVAE
sys.path.remove('/home/wxs/MTVCrafter/')


vqvae_update_config=edict(
    nb_code=4096,
    codebook_dim=2048,
    is_train=False,
)

vision_update_config=edict(
    hrnet_output_level=3,
    vision_guidance_ratio=0,
)

extra_config=edict(
    # vqvae_class=HYBRID_VQVAE,         # a class is not json serializable, will cause error later. so we cannot load it into config dict.
    joint_data_type='joint3d_image_normed',
    resume_path="/home/wxs/LLaMA-Factory/src/llamafactory/extras_byBrad/vqvae_experiment/h36m_j3d_f16s1_cb4096x2048_mpjpe/checkpoint_epoch_47_step_440000",
)


vqvae_config.encoder.out_channels = vqvae_update_config.codebook_dim
vqvae_config.decoder.in_channels = vqvae_update_config.codebook_dim
vqvae_config.vq.nb_code = vqvae_update_config.nb_code
vqvae_config.vq.code_dim = vqvae_update_config.codebook_dim
vqvae_config.vq.is_train = vqvae_update_config.is_train

assert not hasattr(vqvae_config, 'joint_data_type') and not hasattr(vqvae_config, 'resume_path')
setattr(vqvae_config, 'joint_data_type', extra_config.joint_data_type)
setattr(vqvae_config, 'resume_path', extra_config.resume_path)
# setattr(vqvae_config, 'vqvae_class', extra_config.vqvae_class)

vision_config.model.hybrid.hrnet_output_level = vision_update_config.hrnet_output_level
vision_config.model.hybrid.vision_guidance_ratio = vision_update_config.vision_guidance_ratio
