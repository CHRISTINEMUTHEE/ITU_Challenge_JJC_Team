from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import rasterio
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

HEIGHT_NORM_CONSTANT = 30.0


class PixelEmbeddingDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        label_dir: str | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        patch_size: int = 128,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size

        self._train_dataset = None
        self._val_dataset = None
        self._predict_dataset = None

        self.train_size = 0.8
        self.val_size = 0.2

        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str | None = None) -> None:
        label_dir = self.label_dir
        data_dir = self.data_dir

        if stage == "fit":
            if label_dir is None:
                msg = "label_dir must be provided for training"
                raise ValueError(msg)
            file_pairs = _get_file_pairs(data_dir, label_dir)
            dataset = TrainPixelDataset(file_pairs, patch_size=self.patch_size)
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [self.train_size, self.val_size],
                generator=torch.Generator().manual_seed(1994),
            )

        if stage == "predict":
            data_files = sorted(self.data_dir.glob("*.tif"))
            self.predict_dataset = InferencePixelDataset(data_files, patch_size=self.patch_size)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            msg = "train_dataset is not set up"
            raise ValueError(msg)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            msg = "val_dataset is not set up"
            raise ValueError(msg)

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.predict_dataset is None:
            msg = "predict_dataset is not set up"
            raise ValueError(msg)

        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def _get_file_pairs(data_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    """Match tiff files in data_dir with files in label_dir.

    Takes advantage of the fact that the files in a folder all start with the same
    prefix, followed by the numerical id and two-letter code.

    For example:
    data_dir:
        - gee_emb_2013_OG.tif
        - gee_emb_0003_BE.tif
    label_dir:
        - label_0003_BE_2023.tif
        - label_2013_OG_2023.tif
    """
    data_files = sorted(data_dir.glob("*.tif"))
    label_files = sorted(label_dir.glob("*.tif"))

    return list(zip(data_files, label_files, strict=True))


# *****************
# Pixel Datasets
# *****************
class BasePixelDataset(Dataset, ABC):
    def __init__(self, file_paths: list[Path], patch_size: int = 256):
        self.file_paths = file_paths
        self.patch_size = patch_size

    def process_tiff(self, filepath: Path) -> np.ndarray:
        """Load and clean a Tiff file.

        1. Loads
        2. Converts to float32
        3. Replaces NaNs with 0
        4. Pads to patch_size if smaller than patch_size
        """
        with rasterio.open(filepath) as src:
            image = src.read().astype(np.float32)
        image = np.nan_to_num(image)

        _, h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        return image

    def __len__(self):
        return len(self.file_paths)

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class TrainPixelDataset(BasePixelDataset):
    def __init__(self, file_pairs: list[tuple[Path, Path]], patch_size: int = 256):
        emb_paths = [pair[0] for pair in file_pairs]
        super().__init__(emb_paths, patch_size)

        self.file_pairs = file_pairs
        self.rng = np.random.default_rng()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb_path, tar_path = self.file_pairs[idx]

        # Load
        image = self.process_tiff(emb_path)
        target = self.process_tiff(tar_path)

        # Random crop of patch_size x patch_size
        _, h, w = image.shape
        top = self.rng.integers(0, h - self.patch_size + 1)
        left = self.rng.integers(0, w - self.patch_size + 1)
        image = image[:, top : top + self.patch_size, left : left + self.patch_size]
        target = target[:, top : top + self.patch_size, left : left + self.patch_size]

        # Normalize Target Height
        target[3, :, :] = np.clip(target[3, :, :] / HEIGHT_NORM_CONSTANT, 0.0, 1.5)

        return torch.from_numpy(image), torch.from_numpy(target)


class InferencePixelDataset(BasePixelDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        emb_path = self.file_paths[idx]
        image = self.process_tiff(emb_path)

        # Center crop of patch_size x patch_size
        _, h, w = image.shape
        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        image = image[:, top : top + self.patch_size, left : left + self.patch_size]

        # Dummy target for inference
        target = np.zeros_like(image)

        return torch.from_numpy(image), torch.from_numpy(target)
