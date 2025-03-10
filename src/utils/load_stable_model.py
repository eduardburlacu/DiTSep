import os
from hydra.utils import instantiate
from omegaconf import DictConfig
import json
import torch
import pytorch_lightning as pl
from stable_audio_tools.models import create_model_from_config

def load_stable_model(model_config_path:str, ckpt_path:str=None, verbose:bool=False):
    """
    Load a stable model from a checkpoint and model config file.

    Args:
        model_config_path (str): Path to the model config file.
        ckpt_path (str): Path to the checkpoint file.

    Returns:
        model: The loaded model.
    """
    with open(model_config_path) as f:
        model_config = json.load(f)
    
    model = create_model_from_config(model_config)
    
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
    if verbose:
        print(f"Successfully loaded a {model_type} model from the checkpoint: {ckpt_path} with config: {model_config_path}.")
        print(model)
    return model

def load_score_model(model_config:DictConfig, ckpt_path:str=None, verbose:bool=False):
    """
    Load a stable model from a checkpoint and model config file.

    Args:
        model_config (DictConfig): The model config.
        ckpt_path (str): Path to the checkpoint file.

    Returns:
        model: The loaded model.
    """
    score_model = instantiate(model_config, _recursive_=False)
    if ckpt_path:
        ckpt_score = torch.load(ckpt_path)["state_dict"]
        score_model.load_state_dict(ckpt_score, strict=True)
    if verbose:
        print(f"Successfully loaded a {type(score_model)} model from the checkpoint: {ckpt_path}.")
        print(score_model)
    return score_model
