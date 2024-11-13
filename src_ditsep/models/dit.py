"""
Reference: https://github.com/Stability-AI/stable-audio-tools
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/stable_audio_transformer.py
"""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import (
    AttentionProcessor,
    StableAudioAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from diffusers.utils import BaseOutput
from .network_dit import StableAudioGaussianFourierProjection, StableAudioDiTBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "torch.Tensor"  # noqa: F821


class StableAudioDiTModel(ModelMixin, ConfigMixin):
    """
    The Diffusion Transformer model introduced in Stable Audio.

    Parameters:
        sample_size ( `int`, *optional*, defaults to 1024): The size of the input sample.
        in_channels (`int`, *optional*, defaults to 64): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 24): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 24): The number of heads to use for the query states.
        num_key_value_attention_heads (`int`, *optional*, defaults to 12):
            The number of heads to use for the key and value states.
        out_channels (`int`, defaults to 64): Number of output channels.
        cross_attention_dim ( `int`, *optional*, defaults to 768): Dimension of the cross-attention projection.
        time_proj_dim ( `int`, *optional*, defaults to 256): Dimension of the timestep inner projection.
        global_states_input_dim ( `int`, *optional*, defaults to 1536):
            Input dimension of the global hidden states projection.
        cross_attention_input_dim ( `int`, *optional*, defaults to 768):
            Input dimension of the cross-attention projection
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 1024,
        in_channels: int = 64,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        num_key_value_attention_heads: int = 12,
        out_channels: int = 64,
        cross_attention_dim: int = 768,
        time_proj_dim: int = 256,
        global_states_input_dim: int = 1536,
        cross_attention_input_dim: int = 768,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.time_proj = StableAudioGaussianFourierProjection(
            embedding_size=time_proj_dim // 2,
            flip_sin_to_cos=True,
            log=False,
            set_W_to_weight=False,
        )

        self.timestep_proj = nn.Sequential(
            nn.Linear(time_proj_dim, self.inner_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=True),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(global_states_input_dim, self.inner_dim, bias=False),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim, bias=False),
        )

        self.cross_attention_proj = nn.Sequential(
            nn.Linear(cross_attention_input_dim, cross_attention_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cross_attention_dim, cross_attention_dim, bias=False),
        )

        self.preprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.proj_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                StableAudioDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_key_value_attention_heads=num_key_value_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(self.out_channels, self.out_channels, 1, bias=False)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.transformers.hunyuan_transformer_2d.HunyuanDiT2DModel.set_default_attn_processor with Hunyuan->StableAudio
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(StableAudioAttnProcessor2_0())

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        global_hidden_states: torch.FloatTensor = None,
        rotary_embedding: torch.FloatTensor = None,
        return_dict: bool = True,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`StableAudioDiTModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, in_channels, sequence_len)`):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, encoder_sequence_len, cross_attention_input_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            global_hidden_states (`torch.FloatTensor` of shape `(batch size, global_sequence_len, global_states_input_dim)`):
               Global embeddings that will be prepended to the hidden states.
            rotary_embedding (`torch.Tensor`):
                The rotary embeddings to apply on query and key tensors during attention calculation.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token indices, formed by concatenating the attention
                masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            encoder_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_len)`, *optional*):
                Mask to avoid performing attention on padding token cross-attention indices, formed by concatenating
                the attention masks
                    for the two text encoders together. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        cross_attention_hidden_states = self.cross_attention_proj(encoder_hidden_states)
        global_hidden_states = self.global_proj(global_hidden_states)
        time_hidden_states = self.timestep_proj(self.time_proj(timestep.to(self.dtype)))

        global_hidden_states = global_hidden_states + time_hidden_states.unsqueeze(1)

        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        # (batch_size, dim, sequence_length) -> (batch_size, sequence_length, dim)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.proj_in(hidden_states)

        # prepend global states to hidden states
        hidden_states = torch.cat([global_hidden_states, hidden_states], dim=-2)
        if attention_mask is not None:
            prepend_mask = torch.ones((hidden_states.shape[0], 1), device=hidden_states.device, dtype=torch.bool)
            attention_mask = torch.cat([prepend_mask, attention_mask], dim=-1)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    cross_attention_hidden_states,
                    encoder_attention_mask,
                    rotary_embedding,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=cross_attention_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    rotary_embedding=rotary_embedding,
                )

        hidden_states = self.proj_out(hidden_states)

        # (batch_size, sequence_length, dim) -> (batch_size, dim, sequence_length)
        # remove prepend length that has been added by global hidden states
        hidden_states = hidden_states.transpose(1, 2)[:, :, 1:]
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

if __name__ == "__main__":
    from .autoencoder import AutoencoderOobleck
    vae = AutoencoderOobleck()
    dit = StableAudioDiTModel()
    x = torch.rand(8, 1, 8_000 * 10) # bsz 8 of a recording
    print(f"x.shape: {x.shape}")
    z = vae.encode(x).latent_dist.sample()
    print(f"z.shape: {z.shape}")
    z_hat = dit(z)
    print(f"z_hat.shape: {z_hat.shape}")
    x_hat = vae.decode(z_hat.sample)
    print(f"x_hat.shape: {x_hat.shape}")
