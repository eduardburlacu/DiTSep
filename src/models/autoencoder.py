"""
Based on:
    - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_oobleck.py
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_utils import ModelMixin

from models.network_autoencoder import (
    OobleckEncoder,
    OobleckDecoder,
    OobleckDiagonalGaussianDistribution
)
from utils.load_audio import load_audio


@dataclass
class AutoencoderOobleckOutput(BaseOutput):
    """
    Output of AutoencoderOobleck encoding method.

    Args:
        latent_dist (`OobleckDiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and standard deviation of
            `OobleckDiagonalGaussianDistribution`. `OobleckDiagonalGaussianDistribution` allows for sampling latents
            from the distribution.
    """

    latent_dist: "OobleckDiagonalGaussianDistribution"  # noqa: F821


@dataclass
class OobleckDecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, audio_channels, sequence_length)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.Tensor


class AutoencoderOobleck(ModelMixin, ConfigMixin):
    r"""
    An autoencoder for encoding waveforms into latents and decoding latent representations into waveforms. First
    introduced in Stable Audio.

    Parameters:
        encoder_hidden_size (`int`, *optional*, defaults to 128):
            Intermediate representation dimension for the encoder.
        downsampling_ratios (`List[int]`, *optional*, defaults to `[2, 4, 4, 8, 8]`):
            Ratios for downsampling in the encoder. These are used in reverse order for upsampling in the decoder.
        channel_multiples (`List[int]`, *optional*, defaults to `[1, 2, 4, 8, 16]`):
            Multiples used to determine the hidden sizes of the hidden layers.
        decoder_channels (`int`, *optional*, defaults to 128):
            Intermediate representation dimension for the decoder.
        decoder_input_channels (`int`, *optional*, defaults to 64):
            Input dimension for the decoder. Corresponds to the latent dimension.
        audio_channels (`int`, *optional*, defaults to 2):
            Number of channels in the audio data. Either 1 for mono or 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 44100):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
    """
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        encoder_hidden_size = 128, 
        downsampling_ratios=[2, 4, 4, 8, 8],
        channel_multiples=[1, 2, 4, 8, 16],
        decoder_channels=128,
        decoder_input_channels = 64,
        audio_channels=2,
        sampling_rate=8_000, #from the Libri2Mix dataset
    ):
        super().__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.downsampling_ratios = downsampling_ratios
        self.decoder_channels = decoder_channels
        self.upsampling_ratios = downsampling_ratios[::-1]
        self.hop_length = int(np.prod(downsampling_ratios))
        self.sampling_rate = sampling_rate

        self.encoder = OobleckEncoder(
            encoder_hidden_size=encoder_hidden_size,
            audio_channels=audio_channels,
            downsampling_ratios=downsampling_ratios,
            channel_multiples=channel_multiples,
        )

        self.decoder = OobleckDecoder(
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=self.upsampling_ratios,
            channel_multiples=channel_multiples,
        )

        self.use_slicing = False

    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderOobleckOutput, Tuple[OobleckDiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        x = self.pad(x)
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        posterior = OobleckDiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderOobleckOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[OobleckDecoderOutput, torch.Tensor]:
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return OobleckDecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[OobleckDecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.OobleckDecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.OobleckDecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.OobleckDecoderOutput`] is returned, otherwise a plain `tuple`
                is returned.

        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return OobleckDecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[OobleckDecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`OobleckDecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return OobleckDecoderOutput(sample=dec)

    def pad(self, x:torch.Tensor, pad_val:float = 0.0) -> torch.Tensor:
        """
        Pad the input tensor such that it is the closest multiple of the downsampling ratios product.
        """
        x_len = x.shape[-1]
        pad_len = self.hop_length - (x_len % self.hop_length)
        return torch.nn.functional.pad(x, (0, pad_len), value=pad_val)


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
    print(vae)
    print("______________________")
    print(x.shape)
    import pdb; pdb.set_trace()
    posterior = vae.encode(x).latent_dist
    z = posterior.mode()
    print(z.shape)
    x_hat = vae.decode(z).sample
    print(x_hat.shape)
    x_pad = vae.pad(x)
    assert x_pad.shape == x_hat.shape
