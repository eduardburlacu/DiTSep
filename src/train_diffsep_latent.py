import copy
import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm
from pynvml import nvmlInit, nvmlSystemGetDriverVersion


import utils
from datasets import WSJ0_mix_Module, Valentini_Module
from diffsep_latent import LatentDiffSep

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def load_model(config):

    if "score_model" in config.model:
        model_type = "score_model"
        model_obj = LatentDiffSep
    else:
        raise ValueError("config/model should have a score_model sub-config")

    load_pretrained = getattr(config, "load_pretrained", None)
    model = model_obj(config)
    if load_pretrained is not None:
        ckpt_path = Path(to_absolute_path(load_pretrained))

        log.info(f"load pretrained:")
        log.info(f"  - {ckpt_path=}")

        state_dict = torch.load(str(ckpt_path))["state_dict"]

        log.info("Load model state_dict")
        model.load_state_dict(state_dict, strict=True)

    return model, (load_pretrained is not None)


@hydra.main(config_path="./config/latent_diffsep_ouve", config_name="config", version_base=None)
def main(cfg: DictConfig):
    try:
        nvmlInit()
        print("NVML Initialized")
        print("Driver Version:", nvmlSystemGetDriverVersion())
    except Exception as e:
        print("Error initializing NVML:", e)
        # Handle the error or exit the script
        exit(1)

    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CONDA_PREFIX']}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    if utils.ddp.is_rank_zero():
        exp_name = HydraConfig().get().run.dir
        log.info(f"Start experiment: {exp_name}")
    else:
        # when using DDP, if not rank zero, we are already in the run dir
        os.chdir(hydra.utils.get_original_cwd())

    # seed all RNGs for deterministic behavior
    pl.seed_everything(cfg.seed)

    callbacks = []
    # Use a fancy progress bar
    callbacks.append(pl.callbacks.RichProgressBar())
    # configure checkpointing to save all models
    # save_top_k == -1  <-- saves all models
    val_loss_name = f"{cfg.model.main_val_loss}"
    loss_name = val_loss_name.split("/")[-1]  # avoid "/" in filenames
    modelcheckpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=val_loss_name,
        save_top_k=20,
        mode=cfg.model.main_val_loss_mode,
        filename="".join(
            ["epoch-{epoch:03d}_", loss_name, "-{", val_loss_name, ":.3f}"]
        ),
        auto_insert_metric_name=False,
    )
    callbacks.append(modelcheckpoint_callback)

    # the data module
    print("Using the DCASE2020 SELD original dataset")
    log.info("create datalogger")

    if cfg.name == "enhancement":
        dm = Valentini_Module(cfg)
    else:
        dm = WSJ0_mix_Module(cfg)

    # init model
    log.info("Create new model")
    model, is_pretrained = load_model(cfg)

    # create a logger
    if cfg.logger == "wandb":
        pl_logger = pl_loggers.WandbLogger(
            name="OU_Latet_p=0",
            project="diffsep_latent", 
            save_dir=".",
            mode="online" #TODO Change to online when ready
        ) # TODO Change to online when ready
    elif cfg.logger == "tensorboard":
        pl_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="")
    else:
        pl_logger = None
    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints,
    # logs, and more)
    pl_logger.watch(model)

    #trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=tb_logger)
    trainer = pl.Trainer(
        devices=1, #args.num_gpus, [1] for the second GPU, [0, 1] for the first and second GPU etc.
        accelerator="gpu", #"cpu"
        detect_anomaly=False, #
        fast_dev_run=False, # debugging
        num_nodes = 1,
        accumulate_grad_batches=4, 
        precision="16-mixed",
        callbacks=callbacks,
        logger=pl_logger,
        check_val_every_n_epoch=1,
        max_epochs=1_000,
        reload_dataloaders_every_n_epochs = 0 #gradient_clip_val=2.0 #precision="16-mixed",
    )

    if cfg.train:
        log.info("start training")
        ckpt_path = getattr(cfg, "resume_from_checkpoint", None)
        if ckpt_path is None:
    
            trainer.fit(model, dm)

        else:
            trainer.fit(model, dm, ckpt_path=to_absolute_path(ckpt_path))

    if cfg.test:
        try:
            log.info("start testing")
            trainer.test(model, dm, ckpt_path="best")
        except pl.utilities.exceptions.MisconfigurationException:
            log.info(
                "test with current model value because no best model path is available"
            )
            trainer.validate(model, dm)
            trainer.test(model, dm)


if __name__ == "__main__":
    main()
