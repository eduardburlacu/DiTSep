from abc import ABC, abstractmethod
from typing import Any

import torch
import diffusers

import pytorch_lightning as pl

class LDM(ABC):
    def __init__(
            self, 
            encoder_network, 
            score_network,
            scheduler,
            loss,
            ) -> None:
        super().__init__()

class LDMOobleck(LDM, pl.LightningModule):
    """
    A LightningModule for training a Latent Diffusion Model (LDM) on the Oobleck dataset.
    """

    def __init__(self, config: dict):
        super(self, LDMOobleck).__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.sde = None
        self.score_net = None
        self.loss = None

    def forward(self, x):
        return self.diffuser(x)

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.diffuser(x)
        loss = self.loss(z)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=self.config["lr"])
        return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.randn((self.config["batch_size"], self.config["channels"], self.config["time_steps"])),
            batch_size=self.config["batch_size"],
            shuffle=True,
        )