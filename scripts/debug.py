import torch
from models import AutoencoderOobleck
from utils.load_audio import load_audio

if __name__ == '__main__':
    synth = False
    vae = AutoencoderOobleck(
        encoder_hidden_size = 128,
        downsampling_ratios = [2, 4, 4, 8, 8],
        channel_multiples = [1, 2, 4, 8, 16],
        decoder_channels = 128,
        decoder_input_channels = 64,
        audio_channels = 1,
        sampling_rate = 8_000
    )
    if synth:
        x = torch.randn(1, 1, 80_000)
    else:
        path = '/data/milsrg1/corpora/LibriMix/Libri2Mix/wav16k/max/train-100/mix_both/6000-55211-0003_322-124146-0008.wav'
        x = load_audio(path)
        print(x[:,:,:10])
    print("______________________")
    print(x.shape)
    posterior = vae.encode(x).latent_dist
    z = posterior.mode()
    print(z.shape)
    x_hat = vae.decode(z).sample
    print(x_hat.shape)
