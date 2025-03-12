from prefigure.prefigure import push_wandb_config
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import os
import torch
from pynvml import nvmlInit, nvmlSystemGetDriverVersion
import pytorch_lightning as pl
import random

from datasets import WSJ0_mix_Module
from ldm import LDM, LDMDemoCallback

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
    exp_name = HydraConfig().get().run.dir
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

    wandb_logger = pl.loggers.WandbLogger(
        project=cfg.name, 
        mode="online"
    ) # TODO Change to online when ready

    wandb_logger.watch(model)

    exc_callback = ExceptionCallback()

    val_loss_name = f"{cfg.training.main_val_loss}"
    loss_name = val_loss_name.split("/")[-1]  # avoid "/" in filenames
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor=val_loss_name,
        save_top_k=3,
        mode=cfg.training.main_val_loss_mode,
        filename="".join(
            ["epoch-{epoch:03d}_", loss_name, "-{", val_loss_name, ":.3f}"]
        ),
        auto_insert_metric_name=False,
    )

    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
    """
    demo_callback =  LDMDemoCallback(
            demo_dl=train_dl
            demo_every=cfg.training.demo.get("demo_every", 2000), 
            sample_size=cfg.training.demo.sample_size, 
            sample_rate=model_config.fs,
        )
    """

    #Combine args and config dicts
    args_dict = vars(cfg)
    push_wandb_config(wandb_logger, args_dict)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu", #
        detect_anomaly=False, #
        fast_dev_run=False, #
        check_val_every_n_epoch=1,
        num_nodes = 1,
        #precision=getattr(cfg,"precision","16-mixed"),
        callbacks=[ckpt_callback, exc_callback, save_model_config_callback],#demo_callback
        logger=wandb_logger,
        log_every_n_steps=50, 
        max_epochs=1_000,
        reload_dataloaders_every_n_epochs = 0
    )
    trainer.fit(
        model, train_dl, ckpt_path=cfg.ckpt_path if hasattr(cfg,"ckpt_path") else None
    )

if __name__ == '__main__':
    main()
