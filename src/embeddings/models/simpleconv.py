from typing import Any, Literal

import lightning as L
import torch
from torch import nn
from torch.optim import Optimizer


class CNN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        kernel_size: int,
        padding: str | int,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"],
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                n_channels,
                64,
                kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size, padding=padding, padding_mode=padding_mode),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size, padding=padding, padding_mode=padding_mode),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                16,
                n_classes,
                kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if x.shape[-1] != 256 or x.shape[-2] != 256:
        #     x = self.upsample(x)

        x = self.layers(x)
        return x


class SimpleConv(L.LightningModule):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        kernel_size: int,
        padding: str | int = "same",
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "reflect",
    ):
        super().__init__()
        self.model = CNN(n_channels, n_classes, kernel_size, padding, padding_mode)
        self.loss_fn = nn.MSELoss()

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        y_hat = self.model(x)
        return y_hat
