from typing import Any, Literal

import lightning as L
import torch
from torch import nn
from torch.optim import Optimizer

from embeddings.dataloaders import HEIGHT_NORM_CONSTANT


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)

        # mIoU for building, vegetation, water
        for i, name in enumerate(["building", "vegetation", "water"]):
            self.log(f"val_mIoU_{name}", _val_miou(y_hat[:, i], y[:, i]))

        # RMSE for building, vegetation, water (in percentage units)
        for i, name in enumerate(["building", "vegetation", "water"]):
            self.log(f"val_mse_{name}", _val_rmse(y_hat[:, i], y[:, i]))

        # RMSE for height (denormalized to meters)
        y_hat_h = y_hat[:, 3] * HEIGHT_NORM_CONSTANT
        y_h = y[:, 3] * HEIGHT_NORM_CONSTANT
        self.log("val_rmse_height", _val_rmse(y_hat_h, y_h))

        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        y_hat = self.model(x)
        return y_hat


def _val_miou(y_hat: torch.Tensor, y: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    y_hat_bin = y_hat > threshold
    y_bin = y > threshold
    intersection = (y_hat_bin & y_bin).float().sum()
    union = (y_hat_bin | y_bin).float().sum()
    return intersection / (union + 1e-8)


def _val_rmse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((y_hat - y) ** 2).mean()
