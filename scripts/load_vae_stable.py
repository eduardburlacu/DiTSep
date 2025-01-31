import os
import json
import torch
from torch.nn.parameter import Parameter
from stable_audio_tools.models import create_model_from_config

ckpt_path = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "checkpoints", "vae", "vae_finetune.ckpt"
)
model_config = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 
    "stable_audio_tools", 
    "configs", 
    "model_configs", 
    "autoencoders",
    "oobleck_finetune.json"
)


if __name__ == '__main__':


    with open(model_config) as f:
        model_config = json.load(f)
    
    model = create_model_from_config(model_config)
    
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    print(model)

