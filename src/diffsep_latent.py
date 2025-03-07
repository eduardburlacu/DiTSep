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

    def _step(self, y, x):
        x_t, t, sigma, z = self.sample_prior(y, x)
        return self(x_t, t, y), sigma, z

    def compute_score_loss(self, y, x):
        # predict the score
        pred_score, sigma, z = self._step(y, x)
        # compute the MSE loss
        loss = self.loss(pred_score*sigma, -z) #check sign
        loss = loss.mean(dim=tuple(range(2 - loss.ndim , 0)))
        return loss

    def compute_score_loss_init_hack_pit(self, mix, target):
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
        target_latent = target.clone()
        with torch.no_grad(): #pad, encode, normalize the mix and target  
            mix, target_latent = self.encode(mix, target_latent)
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
