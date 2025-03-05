import os
import torch
import torchaudio
import wandb
import pytorch_lightning as pl

from copy import deepcopy
from typing import Optional, Literal

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from stable_audio_tools.models.autoencoders import AudioAutoencoder, fold_channels_into_batch, unfold_channels_from_batch 
from stable_audio_tools.models.bottleneck import VAEBottleneck, RVQBottleneck, DACRVQBottleneck, DACRVQVAEBottleneck, RVQVAEBottleneck, WassersteinBottleneck

from stable_audio_tools.training.losses import MelSpectrogramLoss, MultiLoss, AuralossLoss, ValueLoss, L1Loss, LossWithTarget, MSELoss, HubertLoss
from stable_audio_tools.training.losses import auraloss as auraloss

from stable_audio_tools.training.utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image, log_metric, log_point_cloud, logger_project_name

import utils

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from stable_audio_tools.interface.aeiou import audio_spectrogram_image, tokens_spectrogram_image
from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_model

def trim_to_shortest(a, b):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:,:,:b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:,:,:a.shape[-1]]
    return a, b

class LDM(pl.LightningModule):
    def __init__(
            self,
            config,
            warmup_steps: int = 0,
            warmup_mode: Literal["adv", "full"] = "adv",
            encoder_freeze_on_warmup: bool = False,
            ema_copy = None,
    ):
        super().__init__()
        self.config = config
        
        # Load the pretrained VAE
        self.vae = instantiate(config.model.vae)
        if self.trainable_vae:
            #we add the training wrapper for VAE optimization
            self.vae.requires_grad_(True)
            self.vae.train()
        else:
            self.vae.requires_grad_(False)
            self.vae.eval()
        # Set up EMA for model weights
        self.vae_ema = None if not self.use_ema else EMA(self.vae, **self.config.ema)

        # Warm-up
        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup

        # Optimization 
        self.automatic_optimization = not(
            hasattr(self.config.model, "discriminator") 
            or self.trainable_vae
        )
        self.optimizer_configs = getattr(config.model, "optimizer", None)
        self.loss_config = getattr(self.config.model, "loss", None)

        # Discriminator
        self.use_disc = 'discriminator' in self.loss_config
        self.discriminator = instantiate(self.config.model.discriminator) if self.use_disc else None

        # Losses
        # 1) Adversarial and feature matching losses
        self.gen_loss_modules = torch.nn.ModuleList()
        if self.use_disc:
            if hasattr(self.loss_config,"adv"):
                self.gen_loss_modules.append(instantiate(self.loss_config.adv))
            if hasattr(self.loss_config,"feature_matching"):
                self.gen_loss_modules.append(instantiate(self.loss_config.feature_matching))

        # 2) Spectral Reconstruction loss
        self.gen_loss_modules.append(
            AuralossLoss(self.sdstft, target_key = 'reals', input_key = 'decoded', name='mrstft_loss', weight=self.loss_config['spectral']['weights']['mrstft'], decay = stft_loss_decay)
        )
        if "mrmel" in self.loss_config:
            mrmel_weight = self.loss_config["mrmel"]["weights"]["mrmel"]
            if mrmel_weight > 0:
                mrmel_config = self.loss_config["mrmel"]["config"]
                self.mrmel = MelSpectrogramLoss(self.config.sample_rate,
                    n_mels=mrmel_config["n_mels"],
                    window_lengths=mrmel_config["window_lengths"],
                    pow=mrmel_config["pow"],
                    log_weight=mrmel_config["log_weight"],
                    mag_weight=mrmel_config["mag_weight"],
                )
                self.gen_loss_modules.append(LossWithTarget(
                    self.mrmel, "reals", "decoded",
                    name="mrmel_loss", weight=mrmel_weight,
                ))

        if "hubert" in self.loss_config:
            hubert_weight = self.loss_config["hubert"]["weights"]["hubert"]
            if hubert_weight > 0:
                hubert_cfg = (
                    self.loss_config["hubert"]["config"]
                    if "config" in self.loss_config["hubert"] else dict())
                self.hubert = HubertLoss(weight=1.0, **hubert_cfg)

                self.gen_loss_modules.append(LossWithTarget(
                    self.hubert, target_key = "reals", input_key = "decoded",
                    name="hubert_loss", weight=hubert_weight,
                    decay = self.loss_config["hubert"].get("decay", 1.0)
                ))

        if "l1" in self.loss_config["time"]["weights"]:
            if self.loss_config['time']['weights']['l1'] > 0.0:
                self.gen_loss_modules.append(L1Loss(key_a='reals', key_b='decoded',
                                             weight=self.loss_config['time']['weights']['l1'],
                                             name='l1_time_loss',
                                             decay = self.loss_config['time'].get('decay', 1.0)))

        if "l2" in self.loss_config["time"]["weights"]:
            if self.loss_config['time']['weights']['l2'] > 0.0:
                self.gen_loss_modules.append(MSELoss(key_a='reals', key_b='decoded',
                                             weight=self.loss_config['time']['weights']['l2'],
                                             name='l2_time_loss',
                                             decay = self.loss_config['time'].get('decay', 1.0)))

        if self.vae.bottleneck is not None:
            self.gen_loss_modules += create_loss_modules_from_bottleneck(self.vae.bottleneck)

        self.losses_gen = MultiLoss(self.gen_loss_modules)

        if self.use_disc:
            self.disc_loss_modules = [
                ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss'),
            ]

            self.losses_disc = MultiLoss(self.disc_loss_modules)

        # evaluation losses & metrics
        self.validation_step_outputs = []
        self.val_losses = torch.nn.ModuleDict()
        for name, loss_args in self.config.model.val_losses.items():
            self.val_losses[name] = instantiate(loss_args)
        
    @property
    def use_ema(self):
        return self.config.model.use_ema

    @property
    def trainable_vae(self):
        return self.config.model.vae.trainable_vae
    
    @property
    def lr(self):
        return self.config.model.optimizer.ldm.lr

    @property
    def clip_grad_norm(self):
        return self.config.model.clip_grad_norm

    def configure_optimizers(self):
        gen_params = list(self.vae.parameters())
        # this will be called in on_after_backward
        if self.use_disc:
            opt_gen = create_optimizer_from_config(self.optimizer_configs['ldm']['optimizer'], gen_params)
            opt_disc = create_optimizer_from_config(self.optimizer_configs['discriminator']['optimizer'], self.discriminator.parameters())
            if "scheduler" in self.optimizer_configs['ldm'] and "scheduler" in self.optimizer_configs['discriminator']:
                sched_gen = create_scheduler_from_config(self.optimizer_configs['ldm']['scheduler'], opt_gen)
                sched_disc = create_scheduler_from_config(self.optimizer_configs['discriminator']['scheduler'], opt_disc)
                return [opt_gen, opt_disc], [sched_gen, sched_disc]
            return [opt_gen, opt_disc]
        else:
            opt_gen = create_optimizer_from_config(self.optimizer_configs['ldm']['optimizer'], gen_params)
            if "scheduler" in self.optimizer_configs['ldm']:
                sched_gen = create_scheduler_from_config(self.optimizer_configs['ldm']['scheduler'], opt_gen)
                return [opt_gen], [sched_gen]
            return [opt_gen]

    def forward(self, xt, time, mix):
        return self.score_model(xt, time, mix)

    def training_step(self, batch, batch_idx):
        reals, _ = batch

        log_dict = {}
        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if len(reals.shape) == 2:
            reals = reals.unsqueeze(1)

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        loss_info = {}

        loss_info["reals"] = reals

        encoder_input = reals


        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        if self.warmed_up and self.encoder_freeze_on_warmup:
            with torch.no_grad():
                latents, encoder_info = self.vae.encode(encoder_input, return_info=True)
        else:
            latents, encoder_info = self.vae.encode(encoder_input, return_info=True)

        loss_info["latents"] = latents

        loss_info.update(encoder_info)

        # Encode with teacher model for distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_latents = self.teacher_model.encode(encoder_input, return_info=False)
                loss_info['teacher_latents'] = teacher_latents

        decoded = self.vae.decode(latents)

        #Trim output to remove post-padding
        decoded, reals = trim_to_shortest(decoded, reals)

        loss_info["decoded"] = decoded
        loss_info["reals"] = reals

        if self.vae.out_channels == 2:
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = reals[:, 0:1, :]
            loss_info["reals_right"] = reals[:, 1:2, :]

        # Distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_decoded = self.teacher_model.decode(teacher_latents)
                own_latents_teacher_decoded = self.teacher_model.decode(latents) #Distilled model's latents decoded by teacher
                teacher_latents_own_decoded = self.vae.decode(teacher_latents) #Teacher's latents decoded by distilled model

                loss_info['teacher_decoded'] = teacher_decoded
                loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
                loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded

        if self.use_disc:
            if self.warmed_up:
                loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals=reals, fakes=decoded)
            else:
                loss_adv = torch.tensor(0.).to(reals)
                feature_matching_distance = torch.tensor(0.).to(reals)

                if self.warmup_mode == "adv":
                    loss_dis, _, _ = self.discriminator.loss(reals=reals, fakes=decoded)
                else:
                    loss_dis = torch.tensor(0.0).to(reals)

            loss_info["loss_dis"] = loss_dis
            loss_info["loss_adv"] = loss_adv
            loss_info["feature_matching_distance"] = feature_matching_distance

        opt_gen = None
        opt_disc = None

        if self.use_disc:
            opt_gen, opt_disc = self.optimizers()
        else:
            opt_gen = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        sched_gen = None
        sched_disc = None

        if lr_schedulers is not None:
            if self.use_disc:
                sched_gen, sched_disc = lr_schedulers
            else:
                sched_gen = lr_schedulers

        # Train the discriminator
        use_disc = (
            self.use_disc
            and self.global_step % 2
            # Check warmup mode and if it is time to use discriminator.
            and (
                (self.warmup_mode == "full" and self.warmed_up)
                or self.warmup_mode == "adv")
        )
        if use_disc:
            loss, losses = self.losses_disc(loss_info)

            log_dict['train/disc_lr'] = opt_disc.param_groups[0]['lr']

            opt_disc.zero_grad()
            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad_norm)
            opt_disc.step()

            if sched_disc is not None:
                # sched step every step
                sched_disc.step()

        # Train the generator
        else:

            loss, losses = self.losses_gen(loss_info)

            if self.use_ema:
                self.vae_ema.update()

            opt_gen.zero_grad()
            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.clip_grad_norm)
            opt_gen.step()

            if sched_gen is not None:
                # scheduler step every step
                sched_gen.step()

            log_dict['train/loss'] =  loss.detach().item()
            log_dict['train/latent_std'] = latents.std().detach().item()
            log_dict['train/data_std'] = data_std.detach().item()
            log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach().item()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        reals, _ = batch
        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if len(reals.shape) == 2:
            reals = reals.unsqueeze(1)

        loss_info = {}

        loss_info["reals"] = reals

        encoder_input = reals


        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        with torch.no_grad():
            latents, encoder_info = self.vae.encode(encoder_input, return_info=True)
            loss_info["latents"] = latents
            loss_info.update(encoder_info)

            decoded = self.vae.decode(latents)
            #Trim output to remove post-padding.
            decoded, reals = trim_to_shortest(decoded, reals)

            # Run evaluation metrics.
            val_loss_dict = {}
            for eval_key, eval_fn in self.eval_losses.items():
                loss_value = eval_fn(decoded, reals)
                if eval_key == "sisdr": loss_value = -loss_value
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()

                val_loss_dict[eval_key] = loss_value

        self.validation_step_outputs.append(val_loss_dict)
        return val_loss_dict

    def on_validation_epoch_end(self):
        sum_loss_dict = {}
        for loss_dict in self.validation_step_outputs:
            for key, value in loss_dict.items():
                if key not in sum_loss_dict:
                    sum_loss_dict[key] = value
                else:
                    sum_loss_dict[key] += value

        for key, value in sum_loss_dict.items():
            val_loss = value / len(self.validation_step_outputs)
            val_loss = self.all_gather(val_loss).mean().item()
            log_metric(self.logger, f"val/{key}", val_loss)

        self.validation_step_outputs.clear()  # free memory

    def export_model(self, path, use_safetensors=False):
        if self.vae_ema is not None:
            model = self.vae_ema.ema_model
        else:
            model = self.vae

        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)


class LDMDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        sample_size=65536,
        sample_rate=8000,
        max_demos = 8
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_dl = iter(deepcopy(demo_dl))
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.max_demos = max_demos


    @rank_zero_only
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step
        module.eval()

        try:
            demo_reals, _ = next(self.demo_dl)

            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]

            # Limit the number of demo samples
            if demo_reals.shape[0] > self.max_demos:
                demo_reals = demo_reals[:self.max_demos,...]

            encoder_input = demo_reals
            encoder_input = encoder_input.to(module.device)


            demo_reals = demo_reals.to(module.device)

            with torch.no_grad():
                if module.use_ema:
                    latents = module.vae_ema.ema_model.encode(encoder_input)
                    fakes = module.vae_ema.ema_model.decode(latents)
                else:
                    latents = module.vae.encode(encoder_input)
                    fakes = module.vae.decode(latents)

            #Trim output to remove post-padding.
            fakes, demo_reals = trim_to_shortest(fakes.detach(), demo_reals)
            log_dict = {}

            if module.discriminator is not None:
                window = torch.kaiser_window(512).to(fakes.device)
                fakes_stft = torch.stft(fold_channels_into_batch(fakes), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                fakes_stft.requires_grad = True
                fakes_signal = unfold_channels_from_batch(torch.istft(fakes_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), fakes.shape[1])
                real_stft = torch.stft(fold_channels_into_batch(demo_reals), n_fft=512, hop_length=128, win_length=512, window = window, center=True, return_complex=True)
                reals_signal = unfold_channels_from_batch(torch.istft(real_stft, n_fft=512, hop_length=128, win_length=512, window = window, center=True), demo_reals.shape[1])
                _, loss, _ = module.discriminator.loss(reals_signal,fakes_signal)
                fakes_stft.retain_grad()
                loss.backward()
                grads = unfold_channels_from_batch(fakes_stft.grad.detach().abs(),fakes.shape[1])
                log_dict[f'disciminator_sensitivity'] = wandb.Image(tokens_spectrogram_image(grads.mean(dim=1).log10(), title = 'Discriminator Sensitivity', symmetric = False))
                opts = module.optimizers()
                opts[0].zero_grad()
                opts[1].zero_grad()

            #Interleave reals and fakes
            reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')
            # Put the demos together
            reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')
            
            try:
                data_dir = os.path.join(
                    trainer.logger.save_dir, logger_project_name(trainer.logger),
                    trainer.logger.experiment.id, "media")
                os.makedirs(data_dir, exist_ok=True)
                filename = os.path.join(data_dir, f'recon_{trainer.global_step:08}.wav')
            except:
                filename = f'recon_{trainer.global_step:08}.wav'

            reals_fakes = reals_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, reals_fakes, self.sample_rate)

            log_audio(trainer.logger, 'recon', filename, self.sample_rate)
            log_point_cloud(trainer.logger, 'embeddings_3dpca', latents)
            log_image(trainer.logger, 'embeddings_spec', tokens_spectrogram_image(latents))
            log_image(trainer.logger, 'recon_melspec_left', audio_spectrogram_image(reals_fakes))
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()

def create_loss_modules_from_bottleneck(bottleneck):
    losses = []

    if isinstance(bottleneck, VAEBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        try:
            kl_weight = self.loss_config['bottleneck']['weights']['kl']
        except:
            kl_weight = 1e-6

        kl_loss = ValueLoss(key='kl', weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)

    if isinstance(bottleneck, RVQBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        quantizer_loss = ValueLoss(key='quantizer_loss', weight=1.0, name='quantizer_loss')
        losses.append(quantizer_loss)

    if isinstance(bottleneck, DACRVQBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck):
        codebook_loss = ValueLoss(key='vq/codebook_loss', weight=1.0, name='codebook_loss')
        commitment_loss = ValueLoss(key='vq/commitment_loss', weight=0.25, name='commitment_loss')
        losses.append(codebook_loss)
        losses.append(commitment_loss)

    if isinstance(bottleneck, WassersteinBottleneck):
        try:
            mmd_weight = self.loss_config['bottleneck']['weights']['mmd']
        except:
            mmd_weight = 100

        mmd_loss = ValueLoss(key='mmd', weight=mmd_weight, name='mmd_loss')
        losses.append(mmd_loss)

    return losses


@hydra.main(config_path="./config/ldm", config_name="config", version_base=None)
def main(cfg: DictConfig):
    ldm = LDM(cfg)
    print(ldm)

if __name__ == "__main__":
    main()