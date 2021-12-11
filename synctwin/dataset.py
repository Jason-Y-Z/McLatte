""" 
Dataset and DataLoader designed for SyncTwin.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional


class SyncTwinDataset(Dataset):
    def __init__(
        self, 
        X: np.array,
        M: np.array,
        T: np.array,
        Y_batch: np.array,
        Y_mask: np.array,
    ) -> None:
        super().__init__()

        self._X = X
        self._M = M
        self._T = T
        self._Y_batch = Y_batch
        self._Y_mask = Y_mask
    
    def __len__(self) -> int:
        return self._X.shape[0]
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self._X[idx] * self._M[idx]).float()  # masked covariates
        t = torch.from_numpy(self._T[idx]).float()  # measurement time
        mask = torch.from_numpy(self._M[idx]).float()  # masking vectors
        batch_ind = torch.tensor(idx, dtype=torch.int64)  # batch index
        y_batch = torch.from_numpy(self._Y_batch[idx]).float()  # full outcome
        y_mask = torch.tensor([int(self._Y_mask[idx])]).float()  # outcome mask
        return x, t, mask, batch_ind, y_batch, y_mask


class SyncTwinDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        X: np.array,
        M: np.array,
        T: np.array,
        Y_batch: np.array,
        Y_mask: np.array,
        batch_size: int,
    ):
        super().__init__()

        self._X = X
        self._M = M
        self._T = T
        self._Y_batch = Y_batch
        self._Y_mask = Y_mask
        self._batch_size = batch_size
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            # Train-Validation-Test split
            full_dataset = SyncTwinDataset(
                self._X,
                self._M,
                self._T,
                self._Y_batch,
                self._Y_mask,
            )

            seq_length = len(full_dataset)
            self._train_dataset, self._valid_dataset = random_split(
                full_dataset, 
                [round(seq_length * 0.8), round(seq_length * 0.2)]
            )
    
    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, 
            batch_size=self._batch_size, 
            shuffle=True,
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset, 
            batch_size=self._batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self._valid_dataset, 
            batch_size=self._batch_size,
            num_workers=4,
        )

    def predict_dataloader(self):
        return DataLoader(
            self._valid_dataset, 
            batch_size=self._batch_size,
            num_workers=4,
        )
