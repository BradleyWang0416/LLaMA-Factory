import os
import torch
import numpy as np
from tqdm import tqdm
from llamafactory.extras_byBrad.vqvae import HYBRID_VQVAE
from safetensors.torch import load_file as load_safetensors

def prepare_vqvae(ckpt_path, codebook_dim, nb_code, vision_config, joint_data_type):
    vqvae = HYBRID_VQVAE(vqvae_config.encoder, vqvae_config.decoder, vqvae_config.vq, vision_config=vision_config, joint_data_type=joint_data_type)

    safetensors_path = os.path.join(ckpt_path, "model.safetensors")
    pytorch_bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        print(f"Loading model from {safetensors_path}")
        state_dict = load_safetensors(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_bin_path):
        print(f"Loading model from {pytorch_bin_path}")
        state_dict = torch.load(pytorch_bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in {ckpt_path}")

    vqvae.load_state_dict(state_dict, strict=True)
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False
    vqvae = vqvae.cuda()
    return vqvae

def encode_joint_data(vqvae, joint_data):
    with torch.no_grad():
        codebook_indices, quant_shape = vqvae.encode(joint3d_video=joint_data)
    return codebook_indices.cpu().numpy(), np.array(quant_shape[1:])[None].repeat(quant_shape[0], 0)

def convert_and_save_vqvae_data(dataloader, vqvae, save_dir, joint_data_type):
    os.makedirs(save_dir, exist_ok=True)
    for batch in tqdm(dataloader):
        joint_data = batch[joint_data_type].cuda()
        codebook_indices, quant_shape = encode_joint_data(vqvae, joint_data)

        # Save the codebook indices and quant shape
        sample_id = batch['slice_id'].cpu().numpy()[0]
        np.save(os.path.join(save_dir, f"codebook_indices_{sample_id}.npy"), codebook_indices)
        np.save(os.path.join(save_dir, f"quant_shape_{sample_id}.npy"), quant_shape)

def main(dataloader, ckpt_path, save_dir, joint_data_type, codebook_dim, nb_code, vision_config):
    vqvae = prepare_vqvae(ckpt_path, codebook_dim, nb_code, vision_config, joint_data_type)
    convert_and_save_vqvae_data(dataloader, vqvae, save_dir, joint_data_type)