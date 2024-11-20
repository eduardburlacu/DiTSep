import torch
from safetensors.torch import load_file
from models import AutoencoderOobleck

path = "/data/milsrg1/huggingface/cache/efb48/oobleck_vae/vae.pt" #diffusion_pytorch_model.safetensors

vae = AutoencoderOobleck(
    encoder_hidden_size = 128,
    downsampling_ratios = [2, 4, 4, 8, 8],
    channel_multiples = [1, 2, 4, 8, 16],
    decoder_channels = 128,
    decoder_input_channels = 64,
    audio_channels = 1,
    sampling_rate = 8_000
)

vae.load_state_dict( load_file(path) )
vae.eval()

print(vae)