import os
import math
import torch
import torchaudio
import wandb
import pytorch_lightning as pl

from copy import deepcopy
from typing import Literal

import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
from hydra.utils import instantiate

from stable_audio_tools.models.autoencoders import AudioAutoencoder, fold_channels_into_batch, unfold_channels_from_batch 
from stable_audio_tools.models.bottleneck import VAEBottleneck
from stable_audio_tools.training.losses import MelSpectrogramLoss, MultiLoss, AuralossLoss, ValueLoss, L1Loss, MSELoss, PITLoss
from stable_audio_tools.training.losses import auraloss as auraloss
from stable_audio_tools.training.utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image, log_metric, log_point_cloud, logger_project_name

import sdes
import utils

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from stable_audio_tools.interface.aeiou import audio_spectrogram_image, tokens_spectrogram_image
from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_model
import logging

log = logging.getLogger(__name__)

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
            encoder_freeze_on_warmup: bool = True,
            use_ema: bool = False,
            #ema_copy = None,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        os.environ["HYDRA_FULL_ERROR"] = "1"
        self.use_ema = use_ema
        # Load the VAE and score model
        self.vae = utils.load_stable_model(
            self.config.model.vae.config_path,
            verbose=False,
            )
        log.info(f"Successfully loaded VAE")

        self.score_model = utils.load_score_model(
            self.config.model.score_model.model,
            verbose=False,
        )
        
        log.info("Successfully loaded score model.")

        # Instantiate SDE
        self.sde = instantiate(self.config.model.sde)
        self.t_eps = self.config.model.t_eps
        self.t_max = self.sde.T
        self.t_rev_init = getattr(self.config.model, "t_rev_init", 0.03)
        self.time_sampling_strategy = getattr(
            self.config.model, "time_sampling_strategy", "uniform"
        )
        log.info(f"Sampling time in [{self.t_eps, self.t_max}]")

        # Warm-up
        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup

        # Optimization 
        self.automatic_optimization = False
        self.optimizer_configs = getattr(config.training, "optimizer", None)

        # Discriminator
        self.use_disc = hasattr(self.config.training, "discriminator") and hasattr(self.config.training.loss, "discriminator")
        if self.use_disc:
            self.discriminator = instantiate(self.config.training.discriminator)
        else:
            self.discriminator = None
        
        self.set_train_mode()

        # Losses
        self.loss_config = getattr(config.training, "loss", None)
        if hasattr(self.loss_config, "spectral"):
            sdstft = instantiate(self.loss_config.spectral.config)

        # Adversarial and feature matching losses
        self.gen_loss_modules = []
        if self.use_disc:
            self.gen_loss_modules.append(
                ValueLoss(key='loss_adv', weight=self.loss_config.discriminator.weights.adversarial, name='loss_adv')
            )
            self.gen_loss_modules.append(
                 ValueLoss(key='feature_matching_distance', weight=self.loss_config.discriminator.weights.feature_matching, name='feature_matching_loss')
            )
        stft_loss_decay = self.loss_config.spectral.decay

        self.gen_loss_modules.append(
            PITLoss(
                loss_module=AuralossLoss(
                sdstft, 
                target_key='reals', 
                input_key='decoded', 
                name='mrstft_base', 
                weight=self.loss_config.spectral.weights.mrstft,
                decay=stft_loss_decay
        ),
        input_key='decoded',
        target_key='reals',
        name='pit_mrstft_loss',
        weight=1.0)  
        )
         #AuralossLoss(sdstft, target_key = 'reals', input_key = 'decoded', name='mrstft_loss', weight=self.loss_config.spectral.weights.mrstft, decay = stft_loss_decay)
        if "l1" in self.loss_config.time.weights:
            if self.loss_config.time.weights.l1 > 0.0:
                self.gen_loss_modules.append(PITLoss(
                    loss_module=L1Loss(key_a='reals', key_b='decoded', weight=self.loss_config.time.weights.l1, name='l1_base'),
                    input_key='decoded',
                    target_key='reals',
                    name='pit_l1_loss',
                    weight=1.0
                    )
                )

        if "l2" in self.loss_config.time.weights:
            if self.loss_config['time']['weights']['l2'] > 0.0:
                self.gen_loss_modules.append(PITLoss(
                    MSELoss(key_a='reals', key_b='decoded', weight=self.loss_config.time.weights.l2, name='l2_base'),
                    input_key='decoded',
                    target_key='reals',
                    name='pit_l2_loss',
                    weight=1.0
                    )
                )

        self.losses_gen = MultiLoss(self.gen_loss_modules)

        if self.use_disc:
            self.disc_loss_modules = [
                ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss'),
            ]

            self.losses_disc = MultiLoss(self.disc_loss_modules)

        # evaluation losses & metrics
        self.validation_step_outputs = []
        self.val_losses = dict()
        for name, loss_args in self.config.training.val_losses.items():
            self.val_losses[name] = instantiate(loss_args)

    @property
    def clip_grad_norm(self):
        norm = getattr(self.config.training,"clip_grad_norm", 1.0)
        return norm
    
    @torch.no_grad()
    def encode(self, mix, target):
        mix = utils.pad(mix, self.vae.encoder.hop_length)
        mix = self.vae.encode(mix, iterate_batch=False).unsqueeze(1)
        if target is not None:
            target = utils.pad(target, self.vae.encoder.hop_length)
            bsz, n_src, seq_len = target.shape
            target = target.reshape(bsz*n_src, 1, seq_len)
            target = self.vae.encode(target, iterate_batch=False)
            target = target.reshape(bsz, n_src, *target.shape[1:])
        return mix, target
    
    @torch.no_grad()
    def decode(self, est, target_dim=None):
        bsz, n_src, latent_dim, seq_len = est.shape
        est = est.reshape(bsz * n_src, latent_dim, seq_len)
        est = self.vae.decode(est)
        est = est.reshape(bsz, n_src, -1)
        if target_dim is not None:
            return est[..., :target_dim]
        return est

    def encode_grad(self, mix, target):
        mix = utils.pad(mix, self.vae.encoder.hop_length)
        mix = self.vae.encode(mix, iterate_batch=False).unsqueeze(1)
        self.max_len_lat = max(self.max_len_lat, mix.shape[-1])
        if target is not None:
            target = utils.pad(target, self.vae.encoder.hop_length)
            bsz, n_src, seq_len = target.shape
            target = target.reshape(bsz*n_src, 1, seq_len)
            target = self.vae.encode(target, iterate_batch=False)
            target = target.reshape(bsz, n_src, *target.shape[1:])
        return mix, target
    
    def decode_grad(self, est, target_dim=None):
        bsz, n_src, latent_dim, seq_len = est.shape
        est = est.reshape(bsz * n_src, latent_dim, seq_len)
        est = self.vae.decode(est)
        est = est.reshape(bsz, n_src, -1)
        if target_dim is not None:
            return est[..., :target_dim]
        return est

    @torch.no_grad()
    def separate(self, mix, target_dim = None, latent=False, **kwargs):
        if not latent: #pad the mix to match the VAE input size
            mix, _ = self.encode(mix, None)
        
        sampler_kwargs = self.config.model.sampler.copy()
        with open_dict(sampler_kwargs):
            sampler_kwargs.update(kwargs, merge=True)
        
        #Reverse sampling
        sampler = self.get_pc_sampler(
            "reverse_diffusion", "ald", mix, **sampler_kwargs
        )

        est, *others = sampler()
        est = self.decode(est, target_dim)
        return est, *others

    def separate_grad(self, mix, target_dim = None, latent=False, **kwargs):
        if not latent: #pad the mix to match the VAE input size
            mix, _ = self.encode(mix, None)
        
        sampler_kwargs = self.config.model.sampler.copy()
        with open_dict(sampler_kwargs):
            sampler_kwargs.update(kwargs, merge=True)
        
        #Reverse sampling
        sampler = self.get_pc_sampler(
            "reverse_diffusion", "ald", mix, **sampler_kwargs
        )

        est, *others = sampler()
        est = self.decode_grad(est, target_dim)
        return est, *others

    def set_train_mode(self):
        self.train_encoder = getattr(self.config.model.vae,"train_encoder", False)
        self.train_decoder = getattr(self.config.model.vae,"train_decoder", True)
        self.vae.encoder.requires_grad_(self.train_encoder)
        self.vae.bottleneck.requires_grad_(self.train_encoder)
        self.vae.decoder.requires_grad_(self.train_decoder)
        self.train_score = getattr(self.config.model.score_model,"train_score", False)
        self.score_model.requires_grad_(self.train_score)
        if self.use_disc:
            self.discriminator.requires_grad_(True)
        
    def set_eval_mode(self):
        """Freeze all components for evaluation/testing."""
        self.vae.encoder.requires_grad_(False)
        self.vae.bottleneck.requires_grad_(False)
        self.vae.decoder.requires_grad_(False)
        self.score_model.requires_grad_(False)
        if self.use_disc:
            self.discriminator.requires_grad_(False)

    def sample_time(self, x):
        bsz = x.shape[0]
        return x.new_zeros(bsz).uniform_(self.t_eps, self.t_max)

    def sample_prior(self, mix, target):
        # sample time
        time = self.sample_time(target)
        # get parameters of marginal distribution of x(t)
        mean, std = self.sde.marginal_prob(x0=target, t=time, y=mix)
        # sample normal
        z = torch.randn_like(target)
        # compute x_t
        pad = (...,) + (None,) * (mean.ndim - std.ndim)
        sigma = std[pad] 
        x_t = mean + sigma * z
        return x_t, time, sigma, z

    def forward(self, xt, time, mix):
        return self.score_model(xt, time, mix)

    def on_train_epoch_start(self):
        """Apply training configuration at the start of each training epoch."""
        self.set_train_mode()

    def training_step(self, batch, batch_idx):
        mix, reals = batch
        log_dict = {}
        loss_info = {}

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True
        
        decoded, *_ = self.separate_grad(mix, target_dim=reals.shape[-1], latent=False)

        loss_info["decoded"] = decoded
        loss_info["reals"] = reals
        data_std = reals.std()

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
       
        else: # Train the generator
            loss, losses = self.losses_gen(loss_info)
            if self.use_ema:
                self.autoencoder_ema.update()
            opt_gen.zero_grad()
            self.manual_backward(loss)
            if self.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.vae.decoder.parameters(), self.clip_grad_norm)
            opt_gen.step()
            if sched_gen is not None: # scheduler step every step
                sched_gen.step()

        log_dict['train/loss'] =  loss.detach().item()
        log_dict['train/data_std'] = data_std.detach().item()
        log_dict['train/gen_lr'] = opt_gen.param_groups[0]['lr']
        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach().item()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def on_validation_epoch_start(self):
        """Apply evaluation configuration at the start of validation."""
        self.set_eval_mode()

    def validation_step(self, batch, batch_idx, **kwargs):
        mix, reals = batch        
        decoded, *_ = self.separate(mix, target_dim=reals.shape[-1], latent=False)
        data_std = reals.std()
        loss_info = {}
        loss_info["reals"] = reals
        
        # Run evaluation metrics.
        val_loss_dict = {}
        for eval_key, eval_fn in self.eval_losses.items():
            loss_value = eval_fn(decoded, reals)
            if eval_key == "sisdr": 
                loss_value = -loss_value
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
 
    def test_step(self, batch, batch_idx, **kwargs):
        return self.validation_step(batch, batch_idx, **kwargs)

    def configure_optimizers(self):
        gen_params = list(self.vae.decoder.parameters())

        if self.use_disc:
            opt_gen = create_optimizer_from_config(self.optimizer_configs.generator, gen_params)
            opt_disc = create_optimizer_from_config(self.optimizer_configs.discriminator, self.discriminator.parameters())
            if hasattr(self.optimizer_configs, "scheduler"):
                sched_gen = create_scheduler_from_config(self.optimizer_configs.scheduler, opt_gen)
                sched_disc = create_scheduler_from_config(self.optimizer_configs.scheduler, opt_disc)
                return [opt_gen, opt_disc], [sched_gen, sched_disc]
            return [opt_gen, opt_disc]
        else:
            opt_gen = create_optimizer_from_config(self.optimizer_configs.generator, gen_params)
            if hasattr(self.optimizer_configs, "scheduler"):
                sched_gen = create_scheduler_from_config(self.optimizer_configs.scheduler, opt_gen)
                return [opt_gen], [sched_gen]
            return [opt_gen]

    def get_pc_sampler(self, predictor_name, corrector_name,y,N=None,minibatch=None,schedule=None,**kwargs,):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            if schedule is None:
                return sdes.get_pc_sampler(
                    predictor_name,
                    corrector_name,
                    sde=sde,
                    score_fn=self,
                    y=y,
                    **kwargs,
                )
            else:
                return sdes.get_pc_scheduled_sampler(
                    predictor_name,
                    corrector_name,
                    sde=sde,
                    score_fn=self,
                    y=y,
                    schedule=schedule,
                    **kwargs,
                )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns, intmet = [], [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    if schedule is None:
                        sampler = sdes.get_pc_sampler(
                            predictor_name,
                            corrector_name,
                            sde=sde,
                            score_fn=self,
                            y=y_mini,
                            **kwargs,
                        )
                    else:
                        sampler = sdes.get_pc_scheduled_sampler(
                            predictor_name,
                            corrector_name,
                            sde=sde,
                            score_fn=self,
                            y=y_mini,
                            schedule=schedule,
                            **kwargs,
                        )
                    sample, n, *other = sampler()
                    samples.append(sample)
                    ns.append(n)
                    if len(other) > 0:
                        intmet.append(other[0])
                samples = torch.cat(samples, dim=0)
                if len(intmet) > 0:
                    return samples, ns, intmet
                else:
                    return samples, ns

            return batched_sampling_fn

    def export_model(self, path):
        torch.save(
            {
                "vae": self.vae.state_dict(), 
                "score_model":self.score_model.state_dict()
            }, path)

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

    @property
    def clip_grad_norm(self):
        return self.config.model.clip_grad_norm
