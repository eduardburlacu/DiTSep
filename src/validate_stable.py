from prefigure.prefigure import get_all_args, push_wandb_config
import json
import os
import torch
import pytorch_lightning as pl
import random
import wandb
from stable_audio_tools.datasets import WSJ0_mix_Module
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict

from stable_audio_tools.training.autoencoders import AutoencoderTrainingWrapper

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

def main():
    # List of all checkpoint paths
    checkpoints_dir = os.path.abspath(
        os.path.join(
            os.getcwd(),
            "vae_gan_finetune",
            "vae_gan_finetune",
            "bjpuomso",
            "checkpoints"
        )
    )
    paths = [os.path.join(checkpoints_dir, filename) for filename in os.listdir(checkpoints_dir)]
    checkpoints_dir = os.path.abspath(
        os.path.join(
            os.getcwd(),
            "vae_gan_finetune",
            "vae_gan_finetune",
            "0uyw19a4",
            "checkpoints"
        )
    )
    paths.extend([os.path.join(checkpoints_dir, filename) for filename in os.listdir(checkpoints_dir)])
    paths.sort()
    paths = [paths[4]]

    args = get_all_args()

    seed = args.seed
    dataset_name = "2mix"

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

    datamodule = WSJ0_mix_Module(
        dataset_name = dataset_name, 
        config = dataset_config
    )
    datamodule.setup()
    train_dl = datamodule.train_dataloader()

    wandb_logger = pl.loggers.WandbLogger(
        project=args.name, 
        mode="online"
    ) # TODO Change to online when ready

    exc_callback = ExceptionCallback()
    
    demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    for idx, path in enumerate(paths):
        random.seed(seed)
        torch.manual_seed(seed)
        model = create_model_from_config(model_config)
        copy_state_dict(
            model, load_ckpt_state_dict(path)
        )
    
        if args.remove_pretransform_weight_norm == "pre_load":
            remove_weight_norm_from_model(model.pretransform)

        if args.pretransform_ckpt_path:
            model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

        # Remove weight_norm from the pretransform if specified
        if args.remove_pretransform_weight_norm == "post_load":
            remove_weight_norm_from_model(model.pretransform)

        training_wrapper = create_training_wrapper_from_config(model_config, model, finetune=args.finetune)

        wandb_logger.watch(training_wrapper)
        trainer = pl.Trainer(
            devices=1, #[1]; args.num_gpus
            accelerator="gpu", #"cpu"
            detect_anomaly=False, #
            fast_dev_run=False, #
            num_nodes = args.num_nodes,
            precision=args.precision,
            accumulate_grad_batches=args.accum_batches, 
            callbacks=[demo_callback, exc_callback],
            logger=wandb_logger,
            enable_checkpointing=False,
            log_every_n_steps=1_000, #memory usage vital?!
            max_epochs=1,
            default_root_dir=args.save_dir,
            gradient_clip_val=args.gradient_clip_val,
            reload_dataloaders_every_n_epochs = 0,
        )
        results = trainer.validate(
            training_wrapper, 
            train_dl, 
            ckpt_path=path
        )
                
        wandb_logger.log_table(
            key= "SI-SDR Checkpoints",
            columns = ["Idx","Checkpoint Path","SI-SDR"],
            data=[[idx, path, results]]
        )

        print(f"Results for {path}: {results}")

if __name__ == '__main__':
    main()
