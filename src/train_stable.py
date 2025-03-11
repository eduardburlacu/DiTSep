from prefigure.prefigure import push_wandb_config
import hydra
from omegaconf import DictConfig

import os
import torch
from pynvml import nvmlInit, nvmlSystemGetDriverVersion
import pytorch_lightning as pl
import random

from ldm import LDM
from stable_audio_tools.datasets import WSJ0_mix_Module
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

@hydra.main(config_path="./config/ldm", config_name="config")
def main(cfg:DictConfig):
    try:
        nvmlInit()
        print("NVML Initialized")
        print("Driver Version:", nvmlSystemGetDriverVersion())
    except Exception as e:
        print("Error initializing NVML:", e)
        # Handle the error or exit the script
        exit(1)

    seed = cfg.seed
    dataset_name = "2mix"

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    # seed all RNGs for deterministic behavior
    pl.seed_everything(cfg.seed)

    model_config = cfg.model
    
    datamodule = WSJ0_mix_Module(cfg)

    datamodule.setup()
    train_dl = datamodule.train_dataloader()

    model = LDM.load_from_checkpoint(
        checkpoint_path=cfg.model.score_model.score_ckpt_path, 
        config=cfg, 
        strict=False
    )

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))
    
    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
    
    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    wandb_logger = pl.loggers.WandbLogger(
        project=args.name, 
        mode="offline"
    ) # TODO Change to online when ready
    wandb_logger.watch(training_wrapper)

    exc_callback = ExceptionCallback()
    
    if args.save_dir and isinstance(wandb_logger.experiment.id, str):
        checkpoint_dir = os.path.join(args.save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id, "checkpoints") 
    else:
        checkpoint_dir = None

    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    #Combine args and config dicts
    args_dict = vars(cfg)
    push_wandb_config(wandb_logger, args_dict)

    trainer = pl.Trainer(
        devices=[1],
        accelerator="gpu", #
        detect_anomaly=False, #
        fast_dev_run=False, #
        accumulate_grad_batches=4, 
        check_val_every_n_epoch=1,
        num_nodes = 1,
        precision=cfg.precision,
        accumulate_grad_batches=cfg.training.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=wandb_logger,
        log_every_n_steps=1_000, #memory usage vital?!
        max_epochs=1_000,
        default_root_dir=cfg.save_dir,
        gradient_clip_val=cfg.training.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0
    )
    trainer.fit(training_wrapper, train_dl, ckpt_path=cfg.ckpt_path if cfg.ckpt_path else None)

if __name__ == '__main__':
    main()
