import torch
from diffusers import StableAudioPipeline
from huggingface_hub import hf_hub_download

# Set the cache directory
cache_dir = "/data/milsrg1/huggingface/cache/efb48/"

# Download the VAE model files
vae_model_path = hf_hub_download(
    repo_id="stabilityai/stable-audio-open-1.0",
    filename="vae/diffusion_pytorch_model.safetensors",
    cache_dir=cache_dir
)

vae_config_path = hf_hub_download(
    repo_id="stabilityai/stable-audio-open-1.0",
    filename="vae/config.json",
    cache_dir=cache_dir
)

print(f"Downloaded VAE model to {vae_model_path}")
print(f"Downloaded VAE config to {vae_config_path}")