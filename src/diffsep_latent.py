"""
Combined from:
https://github.com/sp-uhh/sgmse/blob/main/sgmse/model.py
"""

import datetime
import itertools
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import fast_bss_eval
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
from scipy.optimize import linear_sum_assignment
from torch_ema import ExponentialMovingAverage

import sdes
import utils

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class LatentDiffSep(pl.LightningModule):
    def __init__(self, config: DictConfig):
        # init superclass
        super().__init__()

        self.save_hyperparameters()

        # the config and all hyperparameters are saved by hydra to the experiment dir
        self.config = config
        os.environ["HYDRA_FULL_ERROR"] = "1"
        #Instantiate the 3 models
        self.max_len_lat = 0
        self.score_model = instantiate(self.config.model.score_model, _recursive_=False)
        
        vae = utils.load_stable_model(
            self.config.model.vae.config_path,
            self.config.model.vae.ckpt_path,
            verbose=False,
            )
        
        if self.config.model.vae.trainable_vae:
            #we add the training wrapper for VAE optimization
            vae.requires_grad_(True)
            vae.train()
        else:
            vae.requires_grad_(False)
            vae.eval()
        
        self.vae = vae

        self.valid_max_sep_batches = getattr(
            self.config.model, "valid_max_sep_batches", 1
        )
        self.sde = instantiate(self.config.model.sde)
        self.t_eps = self.config.model.t_eps
        self.t_max = self.sde.T
        self.time_sampling_strategy = getattr(
            self.config.model, "time_sampling_strategy", "uniform"
        )
        self.init_hack = getattr(self.config.model, "init_hack", False)
        self.init_hack_p = getattr(self.config.model, "init_hack_p", 1.0 / self.sde.N)
        self.t_rev_init = getattr(self.config.model, "t_rev_init", 0.03)
        log.info(f"Sampling time in [{self.t_eps, self.t_max}]")

        self.lr_warmup = getattr(config.model, "lr_warmup", None)
        self.lr_original = self.config.model.optimizer.lr

        self.train_source_order = getattr(
            self.config.model, "train_source_order", "random"
        )

        # configure the loss functions
        if self.init_hack in [5, 6, 7]:
            if "reduction" not in self.config.model.loss:
                self.loss = instantiate(self.config.model.loss, reduction="none")
            elif self.config.model.loss.reduction != "none":
                raise ValueError("Reduction should 'none' for loss with init_hack == 5")
        else:
            self.loss = instantiate(self.config.model.loss)
        
        self.val_losses = {}
        for name, loss_args in self.config.model.val_losses.items():
            self.val_losses[name] = instantiate(loss_args)

        # for moving average of weights
        self.ema_decay = getattr(self.config.model, "ema_decay", 0.0)
        
        if self.config.model.vae.trainable_vae:
            self.ema = ExponentialMovingAverage(
                self.parameters(), 
                decay=self.ema_decay
            )
        else:
            self.ema = ExponentialMovingAverage(
                self.score_model.parameters(), 
                decay=self.ema_decay
            )

        self._error_loading_ema = False

        self.normalize_batch = utils.normalize_batch
        self.denormalize_batch = utils.denormalize_batch

    @torch.no_grad() #TODO: Change to trainable VAE
    def encode(self, mix, target):
        mix = utils.pad(mix, self.vae.encoder.hop_length)
        target = utils.pad(target, self.vae.encoder.hop_length)
        mix = self.vae.encode(mix, iterate_batch=False).unsqueeze(1)
        self.max_len_lat = max(self.max_len_lat, mix.shape[-1])
        #log.debug(f"Latent Mix shape: {mix.shape}")
        bsz, n_src, seq_len = target.shape
        target = target.reshape(bsz*n_src, 1, seq_len)
        target = self.vae.encode(target, iterate_batch=False)
        target = target.reshape(bsz, n_src, *target.shape[1:])
        return mix, target

    @torch.no_grad() #TODO: Change to trainable VAE
    def decode(self, est):
        bsz, n_src, latent_dim, seq_len = est.shape
        est = est.reshape(bsz * n_src, latent_dim, seq_len)
        est = self.vae.decode(est)
        est = est.reshape(bsz, n_src, -1)
        return est

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
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        return x_t, time, sigma, z

    def forward(self, xt, time, mix):
        return -self.score_model(xt, time, mix)

    def _step(self, y, x):
        x_t, t, sigma, z = self.sample_prior(y, x)
        return self(x_t, t, y), sigma, z

    def compute_score_loss(self, y, x):
        # predict the score
        pred_score, sigma, z = self._step(y, x)
        # compute the MSE loss
        loss = self.loss(pred_score*sigma, -z) #check sign
        if loss.ndim == 4:
            loss = loss.mean(dim=(-3, -2, -1))
        return loss

    def compute_score_loss_with_pit(self, mix, target):

        time = self.sample_time(target)
        # get parameters of marginal distribution of x(t)
        means = []
        for p in itertools.permutations(range(target.shape[1])):
            # get parameters of marginal distribution of x(t)
            mean, std = self.sde.marginal_prob(x0=target[:, p, ...], t=time, y=mix)
            means.append(mean)
        means = torch.stack(means, dim=1)  # (batch, perm, src, latent, seq_len)
        n_perm = means.shape[1]
        
        # sample normal vector
        z = torch.randn_like(target)
        sigma_z = std[:, None, None, None] * z

        # select one of the permutations at random
        mean_select = utils.select_elem_at_random(means, dim=1)
        xt = mean_select + sigma_z[:, None, ...]

        # compute the model mismatch to noise ratio
        err = means - mean_select
        n_elems = (means.shape[1] - 1) * means.shape[2] * means.shape[3] * means.shape[4]
        err_pow = err.square().sum(dim=(1, 2, 3, 4)) / n_elems
        noise_pow = sigma_z.square().mean(dim=(1, 2))
        mmnr = 10.0 * torch.log10(err_pow / noise_pow.clamp(min=1e-5))

        # select which samples require PIT
        select_pit = mmnr < self.config.model.mmnr_thresh_pit
        n_pit = select_pit.sum()
        select_reg = ~select_pit
        n_reg = select_reg.sum()

        losses = []

        # compute loss with pit
        if n_pit > 0:
            mix_ = torch.broadcast_to(
                mix[select_pit, None, ...], (n_pit, n_perm) + mix.shape[-2:]
            )
            mix_ = mix_.flatten(end_dim=1)
            xt_ = torch.broadcast_to(xt[select_pit], (n_pit, n_perm) + xt.shape[-2:])
            xt_ = xt_.flatten(end_dim=1)
            L_ = torch.broadcast_to(
                L[select_pit, None, ...], (n_pit, n_perm) + L.shape[-2:]
            )
            L_ = L_.flatten(end_dim=1)
            z_ = torch.broadcast_to(
                z[select_pit, None, ...], (n_pit, n_perm) + z.shape[-2:]
            )
            z_ = z_.flatten(end_dim=1)
            z_extra = self.sde.mult_std_inv(L_, err[select_pit].flatten(end_dim=1))
            z_pit = z_ + z_extra
            time_ = torch.broadcast_to(time[select_pit, None], (n_pit, n_perm))
            time_ = time_.flatten(end_dim=1)
            pred_pit = self(xt_, time_, mix_)
            loss_pit = (
                (self.sde.mult_std(L_, pred_pit) + z_pit).square().mean(dim=(-2, -1))
            )
            loss_pit = loss_pit.reshape((n_pit, n_perm)).min(dim=-1).values
            losses.append(loss_pit)
            #log.debug(f"Loss PIT shape: {loss_pit.shape}")

        # compute loss without pit
        if n_reg > 0:
            mix_ = mix[select_reg]
            xt_ = xt[select_reg, 0, ...]
            L_ = L[select_reg]
            z_ = z[select_reg]
            pred_reg = self(xt_, time[select_reg], mix_)
            loss_reg = (
                (self.sde.mult_std(L_, pred_reg) + z_).square().mean(dim=(-2, -1))
            )
            losses.append(loss_reg)

        return torch.cat(losses)

    def compute_score_loss_init_hack_pit(self, mix, target):
        """Still thinking what to do here..."""
        # The time is fixed to T here
        time = mix.new_ones(mix.shape[0]) * self.sde.T

        # sample the target noise vector
        z0 = torch.randn_like(target) 

        # we need to recompute mean and L
        losses = []
        for perm in itertools.permutations(range(target.shape[1])):
            mean, std = self.sde.marginal_prob(target[:, perm, ...], time, mix)
            sigma = std[:, None, None, None]

            # include the difference between the real mixture and the model in the noise
            z = z0 + (mix - mean)/sigma
            x_t = mix + sigma * z0
            # predict the score
            pred_score = self(x_t, time, mix)
            # compute the MSE loss
            loss = self.loss(pred_score*sigma, -z).mean(dim=(-3, -2, -1))
            # compute score and error
            losses.append(loss)  # (batch)
            #log.debug(f"Loss init hack shape: {loss.shape}")

        loss_val = torch.stack(losses, dim=1).min(dim=1).values
        return loss_val

    def train_step_init_5(self, mix, target):
        pit = mix.new_zeros(mix.shape[0]).uniform_() < self.init_hack_p
        n_pit = pit.sum()

        losses = []
        if n_pit > 0:
            # loss with pit
            loss_pit = self.compute_score_loss_init_hack_pit(mix[pit], target[pit])
            losses.append(loss_pit)

        if n_pit != mix.shape[0]:
            # loss without pit
            target_nopit = utils.shuffle_sources(target[~pit])
            loss_nopit = self.compute_score_loss(mix[~pit], target_nopit)
            losses.append(loss_nopit)

        # final loss
        loss = torch.cat(losses).mean()

        return loss

    def training_step(self, batch, batch_idx):
        mix, target = batch

        #pad, encode, normalize the mix and target
        mix, target = self.encode(mix, target)

        if self.init_hack == 5:
            loss = self.train_step_init_5(mix, target)
        elif self.train_source_order == "pit":
            loss = self.compute_score_loss_with_pit(mix, target)
        else:
            if self.train_source_order == "power":
                target = utils.power_order_sources(target)
            elif self.train_source_order == "random":
                target = utils.shuffle_sources(target)

            loss = self.compute_score_loss(mix, target)

        # every 10 steps, we log stuff
        cur_step = self.trainer.global_step
        self.last_step = getattr(self, "last_step", 0)
        if cur_step > self.last_step and cur_step % 10 == 0:
            self.last_step = cur_step

            # log the classification metrics
            self.logger.log_metrics(
                {"train/score_loss": loss},
                step=cur_step,
            )
        
        self.do_lr_warmup()

        return loss
    
    def on_validation_epoch_start(self):
        self.n_batches_est_done = 0
    
    def validation_step(self, batch, batch_idx, dataset_i=0):
        mix, target = batch
        
        with torch.no_grad():
            #pad, encode, normalize the mix and target        
            mix, target_latent = self.encode(mix, target)
            (mix, target_latent), *stats = self.normalize_batch((mix, target_latent))
        
            # validation score loss
            if self.init_hack == 5:
                loss = self.train_step_init_5(mix, target_latent)
            else:
                loss = self.compute_score_loss(mix, target_latent)
        del target_latent; torch.cuda.empty_cache()
        self.log("val/score_loss", loss, on_epoch=True, sync_dist=True)
        
        # validation separation losses
        if self.trainer.testing or self.n_batches_est_done < self.valid_max_sep_batches:
            self.n_batches_est_done += 1
            with torch.no_grad():
                est, *_ = self.separate(mix, latent=True, target_dim=target.shape[-1])
                del mix; torch.cuda.empty_cache()

            #log.debug(f"Est shape: {est.shape}, Target shape: {target.shape}")
            for name, loss in self.val_losses.items():
                self.log(name, loss(est, target), on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataset_i=None):
        return self.validation_step(batch, batch_idx, dataset_i=dataset_i)

    def configure_optimizers(self):
        # we may have some frozen layers, so we remove these parameters
        # from the optimization
        log.info(f"set optim with {self.config.model.optimizer}")

        opt_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = instantiate(
            {**{"params": opt_params}, **self.config.model.optimizer}
        )

        if getattr(self.config.model, "scheduler", None) is not None:
            scheduler = instantiate(
                {**self.config.model.scheduler, **{"optimizer": optimizer}}
            )
        else:
            scheduler = None

        # this will be called in on_after_backward
        self.grad_clipper = instantiate(self.config.model.grad_clipper)

        if scheduler is None:
            return [optimizer]
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.config.model.main_val_loss,
            }

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        if self.config.model.vae.trainable_vae:
            self.ema.update(self.parameters())
        else:
            self.ema.update(self.score_model.parameters())

    def on_after_backward(self):
        if self.grad_clipper is not None:
            grad_norm, clipping_threshold = self.grad_clipper(self)
        else:
            # we still want to compute this for monitoring in tensorboard
            grad_norm = utils.grad_norm(self)
            clipped_norm = grad_norm

        # log every few iterations
        if self.trainer.global_step % 25 == 0:
            clipped_norm = min(grad_norm, clipping_threshold)

            # get the current learning reate
            opt = self.trainer.optimizers[0]
            current_lr = opt.state_dict()["param_groups"][0]["lr"]

            self.logger.log_metrics(
                {
                    "grad/norm": grad_norm,
                    "grad/clipped_norm": clipped_norm,
                    "grad/step_size": current_lr * clipped_norm,
                },
                step=self.trainer.global_step,
            )

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        trainable_vae = checkpoint.get("trainable_vae", False)
        if trainable_vae is None:
            log.warning(f"trainable_vae not found in checkpoint! Assuming trainable_vae={self.config.model.vae.trainable_vae}")
        elif  trainable_vae != self.config.model.vae.trainable_vae:
            log.warning(
                f"trainable_vae is set to {trainable_vae} in the checkpoint, but the current model is set to {self.config.model.vae.trainable_vae}")
        
        ema = checkpoint.get("ema", None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self._error_loading_ema = True
            log.warning("EMA state_dict not found in checkpoint!")

    def train(self, mode=True, no_ema=False):
        res = super().train(
            mode
        )  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode is False and not no_ema:
                # eval
                if self.config.model.vae.trainable_vae:
                    self.ema.store(self.parameters())  # store current params in EMA
                    self.ema.copy_to(
                        self.parameters()
                    )  # copy EMA parameters over current params for evaluation
                else:
                    self.ema.store(self.score_model.parameters())  # store current params in EMA
                    self.ema.copy_to(
                        self.score_model.parameters()
                    )
            else:
                # train
                if self.config.model.vae.trainable_vae:
                    if self.ema.collected_params is not None:
                        self.ema.restore(
                            self.parameters()
                        )  # restore the EMA weights (if stored)
                else:
                    if self.ema.collected_params is not None:
                        self.ema.restore(
                            self.score_model.parameters()
                        )
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["trainable_vae"] = self.config.model.vae.trainable_vae
        checkpoint["ema"] = self.ema.state_dict()

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def do_lr_warmup(self):
        if self.lr_warmup is not None and self.trainer.global_step < self.lr_warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.lr_warmup)
            optimizer = self.trainer.optimizers[0]
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr_original

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
        
    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        """
        Use PF-ODE sampler at inference time for the given target y.
        https://github.com/sp-uhh/storm/blob/257e9636a7251ca40aa200753d5c0fe918e31879/sgmse/sampling/__init__.py#L119
        """
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N
        kwargs = {"eps": self.t_eps, **kwargs}
        
        if minibatch is None:
            return sdes.get_ode_sampler(
                sde, self, y=y, **kwargs
            )
        
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sdes.get_ode_sampler(
                        sde, self, y=y_mini, **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    @torch.no_grad()
    def separate(self, mix, target_dim = None,latent=False, **kwargs):
        
        if not latent:
            #pad the mix to match the VAE input size
            mix = utils.pad(mix, self.vae.encoder.hop_length)
            with torch.no_grad():
                mix = self.vae.encode(mix)
        
        sampler_kwargs = self.config.model.sampler.copy()
        with open_dict(sampler_kwargs):
            sampler_kwargs.update(kwargs, merge=True)
        
        #Reverse sampling
        sampler = self.get_pc_sampler(
            "reverse_diffusion", "ald", mix, **sampler_kwargs
        )

        est, *others = sampler()
        #log.debug(f"Est latent shape: {est.shape}")
        est = self.decode(est)
        if target_dim is not None:
            return est[..., :target_dim], *others
        return est, *others

class LatentDiffSep_pp(pl.LightningModule):
    def __init__(self, config: DictConfig):
        # init superclass
        super().__init__()

        self.save_hyperparameters()

        # the config and all hyperparameters are saved by hydra to the experiment dir
        self.config = config
        os.environ["HYDRA_FULL_ERROR"] = "1"
        #Instantiate the 3 models
        
        self.denoiser = instantiate(self.config.model.score_model, _recursive_=False)
        
        vae = utils.load_stable_model(
            self.config.model.vae.config_path,
            self.config.model.vae.ckpt_path,
            verbose=False,
            )
        
        if self.config.model.vae.trainable_vae:
            #we add the training wrapper for VAE optimization
            vae.requires_grad_(True)
            vae.train()
        else:
            vae.requires_grad_(False)
            vae.eval()
        
        self.vae = vae

        self.valid_max_sep_batches = getattr(
            self.config.model, "valid_max_sep_batches", 1
        )
        self.sde = instantiate(self.config.model.sde)
        self.t_eps = self.config.model.t_eps
        self.t_max = self.sde.T
        self.time_sampling_strategy = getattr(
            self.config.model, "time_sampling_strategy", "uniform"
        )
        self.init_hack = getattr(self.config.model, "init_hack", False)
        self.init_hack_p = getattr(self.config.model, "init_hack_p", 1.0 / self.sde.N)
        self.t_rev_init = getattr(self.config.model, "t_rev_init", 0.03)
        log.info(f"Sampling time in [{self.t_eps, self.t_max}]")

        self.lr_warmup = getattr(config.model, "lr_warmup", None)
        self.lr_original = self.config.model.optimizer.lr

        self.train_source_order = getattr(
            self.config.model, "train_source_order", "random"
        )

        # configure the loss functions
        if self.init_hack in [5, 6, 7]:
            if "reduction" not in self.config.model.loss:
                self.loss = instantiate(self.config.model.loss, reduction="none")
            elif self.config.model.loss.reduction != "none":
                raise ValueError("Reduction should 'none' for loss with init_hack == 5")
        else:
            self.loss = instantiate(self.config.model.loss)
        
        self.val_losses = {}
        for name, loss_args in self.config.model.val_losses.items():
            self.val_losses[name] = instantiate(loss_args)

        # for moving average of weights
        self.ema_decay = getattr(self.config.model, "ema_decay", 0.0)
        
        if self.config.model.vae.trainable_vae:
            self.ema = ExponentialMovingAverage(
                self.parameters(), 
                decay=self.ema_decay
            )
        else:
            self.ema = ExponentialMovingAverage(
                self.score_model.parameters(), 
                decay=self.ema_decay
            )

        self._error_loading_ema = False

    def separate(self, mix, **kwargs):
        #pad the mix to match the VAE input size
        mix = utils.pad(mix, self.vae.encoder.hop_length)
        with torch.no_grad():
            mix_latent = self.vae.encode(mix)

        sampler_kwargs = self.config.model.sampler.copy()
        with open_dict(sampler_kwargs):
            sampler_kwargs.update(kwargs, merge=True)
        
        #Reverse sampling
        sampler = self.get_pc_sampler(
            "reverse_diffusion", "ald2", mix_latent, **sampler_kwargs
        )

        est_latent, *others = sampler()

        return self.vae.decode(est_latent)

    def sample_time(self, x):
        return x.new_zeros(x.shape[0]).uniform_(self.t_eps, self.t_max)


    def sample_prior(self, mix, target):
        # sample time
        time = self.sample_time(target)
        # get parameters of marginal distribution of x(t)
        if self.init_hack != 4:
            mean, L = self.sde.marginal_prob(target, time, mix)
        # sample normal vector
        z = torch.randn_like(target)  # (batch, channels, samples)
        true_mix = torch.broadcast_to(mix, target.shape) / target.shape[1]
        if self.init_hack == 1:
            # Simple hack, we replace the mean by the true mixture for times close to T
            # The noise is redefined as z + L^{-1} (mean - true_mix)
            select = time < self.sde.T - self.t_rev_init
            select = torch.broadcast_to(select[:, None, None], z.shape)
            z = torch.where(select, z, z + self.sde.mult_std_inv(L, true_mix - mean))
            # compute x_t
            x_t = mean + self.sde.mult_std(L, z)

        elif self.init_hack == 2:
            # Simple hack, we replace the mean by an interpolated value between
            # mean and the true mixture
            # The noise is left as is
            T = self.sde.T
            Tm = self.sde.T - self.t_rev_init
            beta = torch.clamp((time - Tm) / (T - Tm), min=0.0, max=1.0)
            beta = beta[(...,) + (None,) * (mean.ndim - time.ndim)]
            x_t = true_mix * beta + mean * (1.0 - beta) + self.sde.mult_std(L, z)

        elif self.init_hack == 3:
            # Simple hack, we replace the mean by an interpolated value between
            # mean and the true mixture
            # The noise is redefined as z + L^{-1} (mean - true_mix)
            T = self.sde.T
            Tm = self.sde.T - self.t_rev_init
            beta = torch.clamp((time - Tm) / (T - Tm), min=0.0, max=1.0)
            beta = beta[(...,) + (None,) * (mean.ndim - time.ndim)]

            x_t = true_mix * beta + mean * (1.0 - beta) + self.sde.mult_std(L, z)
            # we want to also learn to predict the mismatch between model and error
            z = self.sde.mult_std_inv(L, x_t - mean)

        elif self.init_hack == 4:
            # We choose a few samples with prop 1 / sde.N and fix the time to 1.0
            select = torch.rand_like(time) < 1 / self.sde.N
            time = torch.where(select, time.new_ones(time.shape) * self.sde.T, time)
            # we need to recompute mean and L
            mean, L = self.sde.marginal_prob(target, time, mix)
            # then we replace the mean by the true mix and redefine the noise to
            # to z + L^{-1} (mean - true_mix) for the modified samples only
            select = torch.broadcast_to(select[:, None, None], z.shape)
            z = torch.where(select, z + self.sde.mult_std_inv(L, true_mix - mean), z)
            # compute x_t
            x_t = mean + self.sde.mult_std(L, z)

        else:
            # compute x_t
            x_t = mean + self.sde.mult_std(L, z)
        return x_t, time, L, z

    def compute_score_loss_with_pit(self, mix, target):
        n_batch = target.shape[0]

        # sample time
        time = self.sample_time(target)

        # get parameters of marginal distribution of x(t)
        means = []
        for p in itertools.permutations(range(target.shape[1])):
            mean, L = self.sde.marginal_prob(target[:, p, :], time, mix)
            means.append(mean)
        means = torch.stack(means, dim=1)  # (batch, perm, src, samples)
        n_perm = means.shape[1]

        # sample normal vector
        z = torch.randn_like(target)  # (batch, channels, samples)
        Lz = self.sde.mult_std(L, z)

        # select one of the permutations at random
        mean_select = utils.select_elem_at_random(means, dim=1)
        xt = mean_select + Lz[:, None, ...]

        # compute the model mismatch to noise ratio
        err = means - mean_select
        n_elems = (means.shape[1] - 1) * means.shape[2] * means.shape[3]
        err_pow = err.square().sum(dim=(1, 2, 3)) / n_elems
        noise_pow = Lz.square().mean(dim=(1, 2))
        mmnr = 10.0 * torch.log10(err_pow / noise_pow.clamp(min=1e-5))

        # select which samples require PIT
        select_pit = mmnr < self.config.model.mmnr_thresh_pit
        n_pit = select_pit.sum()
        select_reg = ~select_pit
        n_reg = select_reg.sum()

        losses = []

        # compute loss with pit
        if n_pit > 0:
            mix_ = torch.broadcast_to(
                mix[select_pit, None, ...], (n_pit, n_perm) + mix.shape[-2:]
            )
            mix_ = mix_.flatten(end_dim=1)
            xt_ = torch.broadcast_to(xt[select_pit], (n_pit, n_perm) + xt.shape[-2:])
            xt_ = xt_.flatten(end_dim=1)
            L_ = torch.broadcast_to(
                L[select_pit, None, ...], (n_pit, n_perm) + L.shape[-2:]
            )
            L_ = L_.flatten(end_dim=1)
            z_ = torch.broadcast_to(
                z[select_pit, None, ...], (n_pit, n_perm) + z.shape[-2:]
            )
            z_ = z_.flatten(end_dim=1)
            z_extra = self.sde.mult_std_inv(L_, err[select_pit].flatten(end_dim=1))
            z_pit = z_ + z_extra
            time_ = torch.broadcast_to(time[select_pit, None], (n_pit, n_perm))
            time_ = time_.flatten(end_dim=1)
            pred_pit = self(xt_, time_, mix_)
            loss_pit = (
                (self.sde.mult_std(L_, pred_pit) + z_pit).square().mean(dim=(-2, -1))
            )
            loss_pit = loss_pit.reshape((n_pit, n_perm)).min(dim=-1).values
            losses.append(loss_pit)

        # compute loss without pit
        if n_reg > 0:
            mix_ = mix[select_reg]
            xt_ = xt[select_reg, 0, ...]
            L_ = L[select_reg]
            z_ = z[select_reg]
            pred_reg = self(xt_, time[select_reg], mix_)
            loss_reg = (
                (self.sde.mult_std(L_, pred_reg) + z_).square().mean(dim=(-2, -1))
            )
            losses.append(loss_reg)

        return torch.cat(losses)

    def compute_score_loss_with_pit_allthetime(self, mix, target):
        """a memory lighter version of the function above (hopefully)"""

        # sample time
        time = self.sample_time(target)

        target = utils.shuffle_sources(target)

        # compute the reference mean
        mean_0, L = self.sde.marginal_prob(target, time, mix)

        # sample the target noise vector
        z0 = torch.randn_like(target)  # (batch, channels, samples)

        # compute x_t
        Lz0 = self.sde.mult_std(L, z0)
        x_t = mean_0 + Lz0  # x_t = mean_0 + Lz0

        # get parameters of marginal distribution of x(t)
        losses = []
        for p in itertools.permutations(range(target.shape[1])):
            # compute the mean for the target permutation
            mean_p, _ = self.sde.marginal_prob(target[:, p, :], time, mix)

            # include the difference between the real mixture and the model in the noise
            z_p = z0 + self.sde.mult_std_inv(L, mean_0 - mean_p)
            # this is the noise if the network decides that the mean is mean_p rather than mean_0
            # i.e. x_t = mean_p + L z_p

            # predict the score
            pred_score = self(x_t, time, mix)

            # compute the MSE loss
            L_score = self.sde.mult_std(L, pred_score)
            loss = self.loss(L_score, -z_p).mean(dim=(-2, -1))

            # compute score and error
            losses.append(loss)  # (batch)

        loss_pit = torch.stack(losses, dim=0).min(dim=0).values

        return loss_pit

    def compute_score_loss_init_hack_pit(self, mix, target):
        """Still thinking what to do here..."""
        # The time is fixed to T here
        time = mix.new_ones(mix.shape[0]) * self.sde.T

        # this is the true target mixture
        true_mix = torch.broadcast_to(mix, target.shape) / target.shape[1]

        # sample the target noise vector
        z0 = torch.randn_like(target)  # (batch, channels, samples)

        # we need to recompute mean and L
        losses = []
        for perm in itertools.permutations(range(target.shape[1])):
            mean, L = self.sde.marginal_prob(target[:, perm, :], time, mix)

            # include the difference between the real mixture and the model in the noise
            z = z0 + self.sde.mult_std_inv(L, true_mix - mean)
            Lz = self.sde.mult_std(L, z)

            # compute x_t
            x_t = mean + Lz

            # predict the score
            pred_score = self(x_t, time, mix)

            # compute the MSE loss
            L_score = self.sde.mult_std(L, pred_score)
            loss = self.loss(L_score, -z).mean(dim=(-2, -1))

            # compute score and error
            losses.append(loss)  # (batch)

        loss_val = torch.stack(losses, dim=1).min(dim=1).values

        return loss_val


    def compute_score_loss(self, mix, target):
        # compute the samples and associated score
        x_t, time, L, z = self.sample_prior(mix, target)
        # predict the score
        pred_score = self(x_t, time, mix)

        # compute the MSE loss
        L_score = self.sde.mult_std(L, pred_score)
        loss = self.loss(L_score, -z)

        if loss.ndim == 3:
            loss = loss.mean(dim=(-2, -1))

        return loss

    def on_train_epoch_start(self):
        pass
    

    @torch.no_grad() #TODO: Change to trainable VAE
    def encode(self, mix, target):
        log.debug("Encoding...")
        log.debug(f"Mix shape: {mix.shape}, Target shape: {target.shape}")
        mix = utils.pad(mix, self.vae.encoder.hop_length)
        target = utils.pad(target, self.vae.encoder.hop_length)
        log.debug(f"Padded Mix shape: {mix.shape}, Padded Target shape: {target.shape}")
        mix = self.vae.encode(mix).unsqueeze(1)
        log.debug(f"Latent Mix shape: {mix.shape}")
        bsz, n_src, seq_len = target.shape
        target = target.view(bsz*n_src, 1, seq_len)
        target = self.vae.encode(target)
        target = target.view(bsz, n_src, *target.shape[1:])
        #log.debug(f"Encoded Target shape(after reshaping): {target.shape}")
        return mix, target

    @torch.no_grad() #TODO: Change to trainable VAE
    def decode(self, est):
        bsz, n_src, latent_dim, seq_len = est.shape
        est = est.view(bsz*n_src, latent_dim, seq_len)
        est = self.vae.decode(est)
        est = est.view(bsz, n_src, -1)
        return est

    def training_step(self, batch, batch_idx):
        mix, target = batch

        #pad, encode, normalize the mix and target
        mix, target = self.encode(mix, target)
        (mix, target), *_ = self.normalize_batch((mix, target))
        
        if self.init_hack == 7:
            loss = self.train_step_init_7(mix, target)
        elif self.init_hack == 6:
            loss = self.train_step_init_6(mix, target)
        elif self.init_hack == 5:
            loss = self.train_step_init_5(mix, target)
        elif self.train_source_order == "pit":
            loss = self.compute_score_loss_with_pit(mix, target)
        else:
            if self.train_source_order == "power":
                target = utils.power_order_sources(target)
            elif self.train_source_order == "random":
                target = utils.shuffle_sources(target)

            loss = self.compute_score_loss(mix, target)

        # every 10 steps, we log stuff
        cur_step = self.trainer.global_step
        self.last_step = getattr(self, "last_step", 0)
        if cur_step > self.last_step and cur_step % 10 == 0:
            self.last_step = cur_step

            # log the classification metrics
            self.logger.log_metrics(
                {"train/score_loss": loss},
                step=cur_step,
            )
        self.do_lr_warmup()

        return loss
 
    def validation_step(self, batch, batch_idx, dataset_i=0):
        mix, target = batch
        
        #pad, encode, normalize the mix and target        
        mix, target = self.encode(mix, target)
        (mix, target), *stats = self.normalize_batch((mix, target))
        
        # validation score loss
        if self.init_hack == 7:
            loss = self.train_step_init_7(mix, target)
        elif self.init_hack == 6:
            loss = self.train_step_init_6(mix, target)
        elif self.init_hack == 5:
            loss = self.train_step_init_5(mix, target)
        else:
            loss = self.compute_score_loss(mix, target)
        self.log("val/score_loss", loss, on_epoch=True, sync_dist=True)

        # validation separation losses
        if self.trainer.testing or self.n_batches_est_done < self.valid_max_sep_batches:
            self.n_batches_est_done += 1
            est, *_ = self.separate(mix)

            est = self.denormalize_batch(est, *stats)

            est = self.decode(est)

            for name, loss in self.val_losses.items():
                self.log(name, loss(est, target), on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self, outputs = None):
        pass

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx, dataset_i=None):
        return self.validation_step(batch, batch_idx, dataset_i=dataset_i)

    def test_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.on_validation_epoch_end(outputs)

    def configure_optimizers(self):
        # we may have some frozen layers, so we remove these parameters
        # from the optimization
        log.info(f"set optim with {self.config.model.optimizer}")

        opt_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = instantiate(
            {**{"params": opt_params}, **self.config.model.optimizer}
        )

        if getattr(self.config.model, "scheduler", None) is not None:
            scheduler = instantiate(
                {**self.config.model.scheduler, **{"optimizer": optimizer}}
            )
        else:
            scheduler = None

        # this will be called in on_after_backward
        self.grad_clipper = instantiate(self.config.model.grad_clipper)

        if scheduler is None:
            return [optimizer]
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.config.model.main_val_loss,
            }

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        if self.config.model.vae.trainable_vae:
            self.ema.update(self.parameters())
        else:
            self.ema.update(self.score_model.parameters())

    def on_after_backward(self):
        if self.grad_clipper is not None:
            grad_norm, clipping_threshold = self.grad_clipper(self)
        else:
            # we still want to compute this for monitoring in tensorboard
            grad_norm = utils.grad_norm(self)
            clipped_norm = grad_norm

        # log every few iterations
        if self.trainer.global_step % 25 == 0:
            clipped_norm = min(grad_norm, clipping_threshold)

            # get the current learning reate
            opt = self.trainer.optimizers[0]
            current_lr = opt.state_dict()["param_groups"][0]["lr"]

            self.logger.log_metrics(
                {
                    "grad/norm": grad_norm,
                    "grad/clipped_norm": clipped_norm,
                    "grad/step_size": current_lr * clipped_norm,
                },
                step=self.trainer.global_step,
            )

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        trainable_vae = checkpoint.get("trainable_vae", False)
        if trainable_vae is None:
            log.warning(f"trainable_vae not found in checkpoint! Assuming trainable_vae={self.config.model.vae.trainable_vae}")
        elif  trainable_vae != self.config.model.vae.trainable_vae:
            log.warning(
                f"trainable_vae is set to {trainable_vae} in the checkpoint, but the current model is set to {self.config.model.vae.trainable_vae}")
        
        ema = checkpoint.get("ema", None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self._error_loading_ema = True
            log.warning("EMA state_dict not found in checkpoint!")

    def train(self, mode=True, no_ema=False):
        res = super().train(
            mode
        )  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode is False and not no_ema:
                # eval
                if self.config.model.vae.trainable_vae:
                    self.ema.store(self.parameters())  # store current params in EMA
                    self.ema.copy_to(
                        self.parameters()
                    )  # copy EMA parameters over current params for evaluation
                else:
                    self.ema.store(self.score_model.parameters())  # store current params in EMA
                    self.ema.copy_to(
                        self.score_model.parameters()
                    )
            else:
                # train
                if self.config.model.vae.trainable_vae:
                    if self.ema.collected_params is not None:
                        self.ema.restore(
                            self.parameters()
                        )  # restore the EMA weights (if stored)
                else:
                    if self.ema.collected_params is not None:
                        self.ema.restore(
                            self.score_model.parameters()
                        )
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["trainable_vae"] = self.config.model.vae.trainable_vae
        checkpoint["ema"] = self.ema.state_dict()

    def _loss(self, forward_out, x_t, z, t, mean, x): pass

    def _step(self, batch, batch_idx):
        mix, target = batch

        #pad, encode, normalize the mix and target
        mix, target = self.encode(mix, target)
        (mix, target), *_ = self.normalize_batch((mix, target))

        t = self.sample_time(target)

        mean, std = self.sde.marginal_prob(target, mix, t)
        z = torch.randn_like(target)
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        return self(x_t, t, mix)

    def do_lr_warmup(self):
        if self.lr_warmup is not None and self.trainer.global_step < self.lr_warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.lr_warmup)
            optimizer = self.trainer.optimizers[0]
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr_original

    def forward(self, xt, t, y):
            F = self.denoiser(self._c_in(t) * xt, self._c_in(t) * y, t)
            # Scaling the network output, see below Eq. (7) in the paper
            if self.network_scaling == "1/sigma":
                std = self.sde._std(t)
                F = F / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                F = F / t[:, None, None, None]

            # The loss type determines the output of the model
            if self.loss_type == "score_matching":
                score = self._c_skip(t) * xt + self._c_out(t) * F
                return score
            elif self.loss_type == "denoiser":
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (F - xt) / sigmas.pow(2)
                return score
            elif self.loss_type == 'data_prediction':
                x_hat = self._c_skip(t) * xt + self._c_out(t) * F
                return x_hat

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_in type: {}".format(self.c_in))
    
    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None] 
        elif self.c_out == "edm":
            sigma = self.sde._std(t)
            return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_out type: {}".format(self.c_out))
    
    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        elif self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_skip type: {}".format(self.c_skip))

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

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
        
    def get_ode_sampler(self, y, N=None, minibatch=1, **kwargs):
        """
        Use PF-ODE sampler at inference time for the given target y.
        https://github.com/sp-uhh/storm/blob/257e9636a7251ca40aa200753d5c0fe918e31879/sgmse/sampling/__init__.py#L119
        """
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N
        kwargs = {"eps": self.t_eps, **kwargs}
        
        if minibatch is None:
            return sdes.get_ode_sampler(
                sde, self, y=y, **kwargs
            )
        
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(math.ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sdes.get_ode_sampler(
                        sde, self, y=y_mini, **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N
        return sdes.get_sb_sampler(sde, self, y=y, sampler_type=sampler_type, **kwargs)
