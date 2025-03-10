import os
import torch
import torchaudio
import wandb
import pytorch_lightning as pl

import itertools
from copy import deepcopy
from typing import Literal

import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
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
            encoder_freeze_on_warmup: bool = False,
            ema_copy = None,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        os.environ["HYDRA_FULL_ERROR"] = "1"

        self.set_train_mode()
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
        
        # PIT-related
        self.init_hack_p = getattr(self.config.model, "init_hack_p", 1.0 / self.sde.N)
        self.train_source_order = getattr(
            self.config.model, "train_source_order", "random"
        )

        # Warm-up
        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.warmup_mode = warmup_mode
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup

        # Discriminator
        self.use_disc = hasattr(self.config.training, "discriminator") 
        self.discriminator = instantiate(self.config.training.discriminator) if self.use_disc else None

        # Optimization 
        self.automatic_optimization = not(
            self.use_disc
            or self.config.model.score_model.train_score
        )
        self.optimizer_configs = getattr(config.training, "optimizer", None)
        self.loss_config = getattr(config.training, "loss", None)
        """
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
        """
        # evaluation losses & metrics
        self.validation_step_outputs = []
        self.val_losses = torch.nn.ModuleDict()
        for name, loss_args in self.config.training.val_losses.items():
            self.val_losses[name] = instantiate(loss_args)

    @torch.no_grad()
    def encode(self, mix, target):
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

    def set_train_mode(self):
        self.train_encoder = getattr(self.config.model.vae,"train_encoder", False)
        self.train_decoder = getattr(self.config.model.vae,"train_decoder", True)
        self.train_score = getattr(self.config.model.score_model,"train_score", False)

    @property
    def train_mode(self):
        """
        The following training mode are possible:
        0: VAE training
        1: Score model training
        2: VAE decoder fine-tunining
        """
        if self.train_encoder and self.train_decoder:
            assert not self.train_score, "Cannot train both VAE and score model at the same time"
            return 0
        elif self.train_decoder:
            assert not self.train_score, "Cannot train both VAE and score model at the same time"
            return 2
        elif self.train_score :
            assert not self.train_encoder and not self.train_decoder, "Cannot train both VAE and score model at the same time"
            return 1
        else:
            raise ValueError("Training mode not recognized")
        
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
    
    def score_loss(self, y, x):
        # predict the score
        x_t, t, sigma, z = self.sample_prior(y, x)
        pred_score = self(x_t, t, y)
        # compute the MSE loss
        loss = self.loss(pred_score*sigma, -z) #check sign
        loss = loss.mean(dim=tuple(range(2 - loss.ndim , 0)))
        return loss
    
    def score_loss_init(self, mix, target):
        # The time is fixed to T here
        time = mix.new_ones(mix.shape[0]) * self.sde.T
        # sample the target noise vector
        z0 = torch.randn_like(target) 
        # we need to recompute mean and L
        losses = []
        pad_dim = (...,) + (None,) * (mix.ndim - time.ndim)
        for perm in itertools.permutations(range(target.shape[1])):
            mean, std = self.sde.marginal_prob(target[:, perm, ...], time, mix)
            sigma = std[pad_dim]
            # include the difference between the real mixture and the model in the noise
            z = z0 + (mix - mean)/sigma
            x_t = mix + sigma * z0
            # predict the score
            pred_score = self(x_t, time, mix)
            # compute the MSE loss
            loss = self.loss(pred_score*sigma, -z)
            loss = loss.mean(dim=tuple(range(2 - loss.ndim , 0)))
            # compute score and error
            losses.append(loss)  # (batch)

        loss_val = torch.stack(losses, dim=1).min(dim=1).values
        return loss_val
    
    def training_step(self, batch, batch_idx):
        if self.global_step >= self.warmup_steps:
            self.warmed_up = True
        
    def validation_step(self, batch, batch_idx, **kwargs):
        return self.val_step_score(
            batch, batch_idx, **kwargs
        ) if self.trainable_score else self.val_step_vae(batch, batch_idx, **kwargs)
 
    def test_step(self, batch, batch_idx, **kwargs):
        return self.validation_step(batch, batch_idx, **kwargs)

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
    def use_ema(self):
        return self.config.model.use_ema

    @property
    def clip_grad_norm(self):
        return self.config.model.clip_grad_norm

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
    ldm = LDM.load_from_checkpoint(
        checkpoint_path=cfg.model.score_model.score_ckpt_path, 
        config=cfg, 
        strict=False
    )
    print(ldm)

if __name__ == "__main__":
    main()