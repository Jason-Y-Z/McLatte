""" 
Dataset and DataLoader designed for representing 
Multi-cause ITE problem setting.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional


class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        X: np.array,
        M: np.array,
        Y_pre: np.array,
        Y_post: np.array,
        A: np.array,
        T: np.array,
    ) -> None:
        super().__init__()

        self._X = X
        self._M = M
        self._Y_pre = Y_pre
        self._Y_post = Y_post
        self._A = A
        self._T = T
    
    def __len__(self) -> int:
        return self._X.shape[0]
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self._X[idx] * self._M[idx]).float()  # masked covariates
        a = torch.from_numpy(self._A[idx]).float()  # treatment
        t = torch.from_numpy(self._T[idx]).float()  # measurement time
        mask = torch.from_numpy(self._M[idx]).float()  # masking vectors
        y_pre = None if self._Y_pre is None else torch.from_numpy(self._Y_pre[idx]).float()  # pre-treatment outcomes
        y_post = torch.from_numpy(self._Y_post[idx]).float()  # post-treatment outcomes
        return x, a, t, mask, y_pre, y_post


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        X: np.array,
        M: np.array,
        Y_pre: np.array,
        Y_post: np.array,
        A: np.array,
        T: np.array,
        batch_size: int,
    ):
        super().__init__()

        self._X = X
        self._M = M
        self._Y_pre = Y_pre
        self._Y_post = Y_post
        self._A = A
        self._T = T
        self._batch_size = batch_size
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            # Train-Validation split
            full_dataset = TimeSeriesDataset(
                self._X,
                self._M,
                self._Y_pre,
                self._Y_post,
                self._A,
                self._T,
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