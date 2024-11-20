<<<<<<< HEAD
=======
<<<<<<< HEAD:src_ditsep/train_vae.py
from prefigure.prefigure import get_all_args, push_wandb_config
import json
import os
import torch
import pytorch_lightning as pl
import random

from stable_audio_tools.data.dataset import create_dataloader_from_config
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

def main():

    args = get_all_args()

    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    random.seed(seed)
    torch.manual_seed(seed)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    model = create_model_from_config(model_config)

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

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
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
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    push_wandb_config(wandb_logger, args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2, 
                                        contiguous_gradients=True, 
                                        overlap_comm=True, 
                                        reduce_scatter=True, 
                                        reduce_bucket_size=5e8, 
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True
                                        )
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto" 

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0
    )

    trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()
=======
>>>>>>> db0c0f8 (rebase solved)
import os
import sys

import copy
from logging import getLogger

from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

#---Insert main project directory so that we can resolve the src imports---
src_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, src_path)
import utils
from .datasets import WSJ0_mix_Module, Valentini_Module
from .models import AutoencoderOobleck

log = getLogger(__name__)


def load_model(config):

    if "score_model" in config.model:
        model_type = "score_model"
        model_obj = DiffSepModel
    else:
        raise ValueError("config/model should have a score_model sub-config")

    load_pretrained = getattr(config, "load_pretrained", None)
    if load_pretrained is not None:
        ckpt_path = Path(to_absolute_path(load_pretrained))
        hparams_path = (
            ckpt_path.parents[1] / "hparams.yaml"
        )  # path when using lightning checkpoint
        hparams_path_alt = (
            ckpt_path.parents[0] / "hparams.yaml"
        )  # path when using calibration output checkpoint

        log.info(f"load pretrained:")
        log.info(f"  - {ckpt_path=}")

        if hparams_path_alt.exists():
            log.info(f"  - {hparams_path_alt=}")
            # this was produced by the calibration routing
            with open(hparams_path, "r") as f:
                conf = yaml.safe_load(f)
                config_seld_model = conf["config"]["model"][model_type]

            config.model.seld_model.update(config_seld_model)
            model = model_obj(config)

            state_dict = torch.load(str(ckpt_path))

            log.info("Load model state_dict")
            model.load_state_dict(state_dict, strict=True)

        elif hparams_path.exists():
            log.info(f"  - {hparams_path=}")
            # this is a checkpoint
            with open(hparams_path, "r") as f:
                conf = yaml.safe_load(f)
                config_seld_model = conf["config"]["model"][model_type]

            config.model.seld_model.update(config_seld_model)

            log.info("Load model from lightning checkpoint")
            model = model_obj.load_from_checkpoint(
                ckpt_path, strict=True, config=config
            )

        else:
            raise ValueError(
                f"Could not find the hparams.yaml file for checkpoint {ckpt_path}"
            )

    else:
        model = model_obj(config)

    return model, (load_pretrained is not None)


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    log.info(OmegaConf.to_yaml(cfg))
    if utils.ddp.is_rank_zero():
        exp_name = HydraConfig().get().run.dir
        log.info(f"Start experiment: {exp_name}")
    else:
        # when using DDP, if not rank zero, we are already in the run dir
        os.chdir(hydra.utils.get_original_cwd())

    # seed all RNGs for deterministic behavior
    pl.seed_everything(cfg.seed)

    torch.autograd.set_detect_anomaly(True)

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
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=".", name="", version="")

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints,
    # logs, and more)
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=tb_logger)

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
<<<<<<< HEAD
=======
>>>>>>> 4f83de4 (codebase with models proposed):src/train_vae.py
>>>>>>> db0c0f8 (rebase solved)
