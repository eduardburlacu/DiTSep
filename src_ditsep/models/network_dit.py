from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    StableAudioAttnProcessor2_0,
)

class StableAudioGaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    # Copied from diffusers.models.embeddings.GaussianFourierProjection.__init__
    def __init__(
            self,
            embedding_size: int = 256,
            scale: float = 1.0,
            set_W_to_weight=True,
            log=True,
            flip_sin_to_cos=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

        if set_W_to_weight:
            # to delete later
            del self.weight
            self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
            self.weight = self.W
            del self.W

    def forward(self, x):
        if self.log:
            x = torch.log(x)

        x_proj = 2 * np.pi * x[:, None] @ self.weight[None, :]

        if self.flip_sin_to_cos:
            out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out


@maybe_allow_in_graph
class StableAudioDiTBlock(nn.Module):
    r"""
    Transformer block used in Stable Audio model (https://github.com/Stability-AI/stable-audio-tools). Allow skip
    connection and QKNorm

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for the query states.
        num_key_value_attention_heads (`int`): The number of heads to use for the key and value states.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_key_value_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        upcast_attention: bool = False,
        norm_eps: float = 1e-5,
        ff_inner_dim: Optional[int] = None,
    ):
        super().__init__()
        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            out_bias=False,
            processor=StableAudioAttnProcessor2_0(),
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, norm_eps, True)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            kv_heads=num_key_value_attention_heads,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            out_bias=False,
            processor=StableAudioAttnProcessor2_0(),
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, True)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn="swiglu",
            final_dropout=False,
            inner_dim=ff_inner_dim,
            bias=True,
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rotary_embedding: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_embedding,
        )

        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states

        return hidden_states

if __name__ == '__main__':
    # Define the model
    model = StableAudioDiTBlock(
        dim=256,
        num_attention_heads=16,
        num_key_value_attention_heads=8,
        attention_head_dim=32
    )
    # Set the model to evaluation mode
    model.eval()
    # Generate a random input tensor
    x = torch.rand(4, 256)
    # Run the model
    with torch.no_grad():
        output = model(x)
    # Print the output shape
    print(output.shape)
