from typing import Optional
import math
import torch
from torch import nn
from torch.nn.utils import weight_norm
from diffusers.utils.torch_utils import randn_tensor

class Snake1d(nn.Module):
    """
    A 1-dimensional Snake activation function module.
    """

    def __init__(self, hidden_dim, logscale=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))

        self.alpha.requires_grad = True
        self.beta.requires_grad = True
        self.logscale = logscale

    def forward(self, hidden_states):
        shape = hidden_states.shape

        alpha = self.alpha if not self.logscale else torch.exp(self.alpha)
        beta = self.beta if not self.logscale else torch.exp(self.beta)

        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        hidden_states = hidden_states.reshape(shape)
        return hidden_states


class OobleckResidualUnit(nn.Module):
    """
    A residual unit composed of Snake1d and weight-normalized Conv1d layers with dilations.
    """

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2

        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, hidden_state):
        """
        Forward pass through the residual unit.

        Args:
            hidden_state (`torch.Tensor` of shape `(batch_size, channels, time_steps)`):
                Input tensor .

        Returns:
            output_tensor (`torch.Tensor` of shape `(batch_size, channels, time_steps)`)
                Input tensor after passing through the residual unit.
        """
        output_tensor = hidden_state
        output_tensor = self.conv1(self.snake1(output_tensor))
        output_tensor = self.conv2(self.snake2(output_tensor))

        padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        output_tensor = hidden_state + output_tensor
        return output_tensor


class OobleckEncoderBlock(nn.Module):
    """Encoder block used in Oobleck encoder."""

    def __init__(self, input_dim, output_dim, stride: int = 1):
        super().__init__()

        self.res_unit1 = OobleckResidualUnit(input_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(input_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(input_dim, dilation=9)
        self.snake1 = Snake1d(input_dim)
        self.conv1 = weight_norm(
            nn.Conv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        )

    def forward(self, hidden_state):
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.snake1(self.res_unit3(hidden_state))
        hidden_state = self.conv1(hidden_state)

        return hidden_state


class OobleckDecoderBlock(nn.Module):
    """Decoder block used in Oobleck decoder."""

    def __init__(self, input_dim, output_dim, stride: int = 1):
        super().__init__()

        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state):
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.res_unit3(hidden_state)

        return hidden_state


class OobleckDiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.scale = parameters.chunk(2, dim=1)
        self.std = nn.functional.softplus(self.scale) + 1e-4
        self.var = self.std * self.std
        self.logvar = torch.log(self.var)
        self.deterministic = deterministic

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "OobleckDiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return (self.mean * self.mean + self.var - self.logvar - 1.0).sum(1).mean()
            else:
                normalized_diff = torch.pow(self.mean - other.mean, 2) / other.var
                var_ratio = self.var / other.var
                logvar_diff = self.logvar - other.logvar

                kl = normalized_diff + var_ratio + logvar_diff - 1

                kl = kl.sum(1).mean()
                return kl

    def mode(self) -> torch.Tensor:
        return self.mean

class OobleckEncoder(nn.Module):
    """Oobleck Encoder"""

    def __init__(self, encoder_hidden_size, audio_channels, downsampling_ratios, channel_multiples):
        super().__init__()

        strides = downsampling_ratios
        channel_multiples = [1] + channel_multiples

        # Create first convolution
        self.conv1 = weight_norm(nn.Conv1d(audio_channels, encoder_hidden_size, kernel_size=7, padding=3))

        self.block = []
        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride_index, stride in enumerate(strides):
            self.block += [
                OobleckEncoderBlock(
                    input_dim=encoder_hidden_size * channel_multiples[stride_index],
                    output_dim=encoder_hidden_size * channel_multiples[stride_index + 1],
                    stride=stride,
                )
            ]

        self.block = nn.ModuleList(self.block)
        d_model = encoder_hidden_size * channel_multiples[-1]
        self.snake1 = Snake1d(d_model)
        self.conv2 = weight_norm(nn.Conv1d(d_model, encoder_hidden_size, kernel_size=3, padding=1))

    def forward(self, hidden_state):
        hidden_state = self.conv1(hidden_state)

        for module in self.block:
            hidden_state = module(hidden_state)

        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv2(hidden_state)

        return hidden_state


class OobleckDecoder(nn.Module):
    """Oobleck Decoder"""

    def __init__(self, channels, input_channels, audio_channels, upsampling_ratios, channel_multiples):
        super().__init__()

        strides = upsampling_ratios
        channel_multiples = [1] + channel_multiples

        # Add first conv layer
        self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3))

        # Add upsampling + MRF blocks
        block = []
        for stride_index, stride in enumerate(strides):
            block += [
                OobleckDecoderBlock(
                    input_dim=channels * channel_multiples[len(strides) - stride_index],
                    output_dim=channels * channel_multiples[len(strides) - stride_index - 1],
                    stride=stride,
                )
            ]

        self.block = nn.ModuleList(block)
        output_dim = channels
        self.snake1 = Snake1d(output_dim)
        self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def forward(self, hidden_state):
        hidden_state = self.conv1(hidden_state)

        for layer in self.block:
            hidden_state = layer(hidden_state)

        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv2(hidden_state)

        return hidden_state