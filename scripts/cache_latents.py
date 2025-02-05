import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..src.datasets.wsj0_mix import WSJ0_mix, WSJ0_mix_Module
from ..src.stable_audio_tools.models import create_model_from_config
from ..src.utils.load_audio import load_audio  


# Define paths
ckpt_path = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "checkpoints", "vae", "vae_finetune.ckpt"
)
model_config = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 
    "src",
    "stable_audio_tools", 
    "configs", 
    "model_configs", 
    "autoencoders",
    "oobleck_finetune.json"
)

output_path = "/research/milsrg1/user_workspace/efb48/latents"
os.makedirs(output_path, exist_ok=True)

# Load the dataset
dataset = WSJ0_mix(
    path=dataset_path, 
    n_spkr=2, 
    fs=8_000, 
    cut="max", 
    split="train"
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load the VAE model
with open(model_config) as f:
    model_config = json.load(f)

vae = create_model_from_config(model_config)
ckpt = torch.load(ckpt_path)
vae.load_state_dict(ckpt['state_dict'])
vae.eval()

# Process and save latent features
for i, (mix, tgt) in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        mix_latent = vae.encode(mix).latent_dist.mode()
        tgt_latent = vae.encode(tgt).latent_dist.mode()

    # Save the latent features
    torch.save(mix_latent, os.path.join(output_path, f"mix_latent_{i}.pt"))
    torch.save(tgt_latent, os.path.join(output_path, f"tgt_latent_{i}.pt"))

print("Latent features saved successfully.")
