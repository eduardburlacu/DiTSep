from prefigure.prefigure import push_wandb_config
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import os
import torch
from torch.utils.data import DataLoader
from pynvml import nvmlInit, nvmlSystemGetDriverVersion
import pytorch_lightning as pl
import random

from datasets import WSJ0_mix_Module, WSJ0LatentDataset, device_aware_max_collator
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
    model = LDM.load_from_checkpoint(
        checkpoint_path=cfg.model.score_model.score_ckpt_path, 
        config=cfg, 
        strict=False
    )

    datamodule = WSJ0_mix_Module(cfg)
    datamodule.setup()
    val_dataloader = datamodule.val_dataloader()

    # Setup latent caching
    use_latent_cache = getattr(cfg, 'use_latent_cache', True)
    latent_cache_dir = cfg.get('latent_cache_dir', os.path.join(os.getcwd(), "cached_latents"))
    num_samples_per_mixture = cfg.get('num_samples_per_mixture', 6)
    
    # Get original dataset
    train_dataset = datamodule.datasets["librimix_train-360"]
    
    # Check if we need to generate latent cache
    metadata_path = os.path.join(latent_cache_dir, "metadata.pt")
    if use_latent_cache and not os.path.exists(metadata_path):
        print(f"Generating latent cache in {latent_cache_dir}...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create dataloader with batch_size=1 and shuffle=False for latent generation
        gen_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.get('num_workers', 4),
            collate_fn=lambda batch: device_aware_max_collator(batch, None)  # Keep on CPU, will move in generate_dataset
        )
        
        # Generate latent cache
        model.generate_dataset(
            gen_dataloader, 
            output_dir=latent_cache_dir,
            num_samples_per_mixture=num_samples_per_mixture
        )
        print("Latent cache generation completed.")
    
    # Create train dataloader
    if use_latent_cache and os.path.exists(metadata_path):
        print(f"Using latent cache from {latent_cache_dir}")
        
        # Create latent dataset with the correct original dataset
        latent_dataset = WSJ0LatentDataset(
            latent_cache_dir,
            train_dataset
        )
        
        # Create latent dataloader with max_collator
        train_dl = DataLoader(
            latent_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.get('num_workers', 4),
            pin_memory=True,
            collate_fn= lambda batch: device_aware_max_collator(batch, device) 
        )
        print(f"Created latent dataloader with {len(latent_dataset)} samples")
    else:
        # Use original dataloader
        train_dl = datamodule.train_dataloader()
        print("Using original WSJ0_mix dataloader")

    
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
        callbacks=[ckpt_callback, exc_callback, save_model_config_callback],#demo_callback
        logger=wandb_logger,
        log_every_n_steps=50, 
        max_epochs=1_000,
        reload_dataloaders_every_n_epochs = 0
    )

    trainer.fit(
        model, 
        train_dataloaders=train_dl, 
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.ckpt_path if hasattr(cfg,"ckpt_path") else None
    )

if __name__ == '__main__':
    main()
