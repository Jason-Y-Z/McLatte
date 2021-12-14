""" 
Dataset and Dataloader designed for time series predictions.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional


class ShiftingDataset(Dataset):
    """
    This dataset takes an input time series, shifting the timestep by 1, 
    and use that as the target for prediction.
    """
    def __init__(self, Y: np.array, seq_len: int) -> None:
        super().__init__()

        self._Y = Y
        self._n_subject = Y.shape[0]
        self._seq_len = seq_len
    
    def __len__(self) -> int:
        return (self._Y.shape[1] - self._seq_len) * self._n_subject
    
    def __getitem__(self, idx):
        """
        The indexing enumerates the subject dimension first 
        and the time dimension second.
        """
        time_start = idx // self._n_subject
        time_end = time_start + self._seq_len
        subject_idx = idx % self._n_subject
        return (
            torch.from_numpy(
                self._Y[subject_idx, time_start:time_end]
            ).unsqueeze(1).float(),
            torch.from_numpy(
                self._Y[subject_idx, time_start + 1:time_end + 1]
            ).unsqueeze(1).float()
        )


class ShiftingDataModule(pl.LightningDataModule):
    def __init__(self, Y: np.array, seq_len: int, batch_size: int):
        super().__init__()

        self._Y = Y
        self._seq_len = seq_len
        self._batch_size = batch_size
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            full_dataset = ShiftingDataset(
                self._Y,
                self._seq_len,
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
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset, 
            batch_size=self._batch_size,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self._valid_dataset, 
            batch_size=self._batch_size,
            num_workers=4,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self._valid_dataset, 
            batch_size=self._batch_size,
            num_workers=4,
            persistent_workers=True,
        )
