from typing import Literal
from dataclasses import dataclass
from importlib import import_module

import torch
import torch.nn as nn
import torch.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_utils import ModelMixin

@dataclass
class SeparatorBuilder:
    """
    Configuration for DiTSep model
    """
    vae: Literal['vae', 'facodec'] = 'vae'
    score_model: Literal['NCSN++', "DiT"]
    sde: Literal["OU_VE", "DiffSep"]
    
    def instantiate_vae(self):
        if self.vae == 'vae':
            OobleckVAE = import_module('models.vae').OobleckVAE
            vae = OobleckVAE(
                encoder_hidden_size = 128, 
                downsampling_ratios=[2, 4, 4, 8, 8],
                channel_multiples=[1, 2, 4, 8, 16],
                decoder_channels=128,
                decoder_input_channels = 64,
                audio_channels=1,
                sampling_rate=8_000,
            )

        elif self.vae == 'facodec':
            FACodec = import_module('models.facodec.facodec').FACodec
            vae = FACodec(
                ngf = 32, 
                in_channels = 256, 
                out_channels = 256, 
                up_ratios = [2, 4, 5, 5],
                codebook = None,
                gr = None,
            )
        else:
            raise ValueError("Invalid VAE")
        return vae
    
    def instantiate_score_model(self):
        if self.score_model == 'NCSN++':
            
            NCSNpp = import_module('models.ncsnpp').NCSNpp
            score_model = NCSNpp(
                in_channels=1,
                out_channels=1,
                nf=128,
                n_res_blocks=4,
                resamp_with_conv=True,
                downscale_method='conv',
                dropout_rate=0.0,
                ncsnpp=True,
            )
        elif self.score_model == 'DiT':
            DiT = import_module('models.dit').DiT
            score_model = DiT(
                dim = 256,
                depth = 6,
                heads = 8,
                mlp_dim = 512,
                channels = 1,
                dim_head = 64,
                dropout = 0.0,
                emb_dropout = 0.0,
                num_classes = 256,
                pool = 'cls',
                dim_out = 256,
            )
        else:
            raise ValueError("Invalid score model")
        return score_model

    def instantiate_sde(self):
        if sde == 'OU_VE':
            OU_VE = import_module('sdes.ou_ve').OU_VE
            sde = OU_VE()
        elif sde == 'DiffSep':
            DiffSep = import_module('sdes.diffsep').DiffSep
            sde = DiffSep()
        else:
            raise ValueError("Invalid SDE")
        return sde

    def build(self):
        vae = self.instantiate_vae()
        score_model = self.instantiate_score_model()
        sde = self.instantiate_sde()
        return vae, score_model, sde
    

class DiTSep(
    ModelMixin, 
    ConfigMixin
    ):
    """
    DiTSep model for Blind Source Separation
    Parameters:

    """
    
    @register_to_config()
    def __init__(
        self,
        vae,
        score_model,
        grad_clipper,
        max_seq_len: int = 247_808,
        in_channels: int = 1,
        out_channels: int = 2,
        sample_rate: int = 8_000,
        ):
        super().__init__()
        self.vae = vae
        self.score_model = score_model
        self.grad_clipper = grad_clipper
