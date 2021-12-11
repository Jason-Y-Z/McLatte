""" 
Baseline RNN model for time series predictions.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from .dataset import ShiftingDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from typing import Dict


RNNS = {
    'rnn': torch.nn.RNN,
    'lstm': torch.nn.LSTM,
    'gru': torch.nn.GRU,
}


class BaselineRnn(pl.LightningModule):
    def __init__(
        self, 
        rnn: nn.Module,
        hidden_dim: int,
        output_dim: int,
        lr: float,
        gamma: float,
    ):
        super().__init__()

        self._rnn = rnn
        self._linear = nn.Linear(hidden_dim, output_dim)
        self._lr = lr
        self._gamma = gamma
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=self._gamma
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    def forward(self, y):
        output, _ = self._rnn(torch.transpose(y, 0, 1))
        y_tilde = torch.transpose(self._linear(output), 0, 1)
        return y_tilde
    
    def training_step(self, batch, batch_idx):
        y, y_shifted = batch

        y_tilde = self(y)
        loss = torch.nn.functional.mse_loss(y_tilde, y_shifted)

        self.log('ptl/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y, y_shifted = batch

        y_tilde = self(y)
        loss = torch.nn.functional.l1_loss(y_tilde, y_shifted)

        self.log('ptl/valid_loss', loss)
        return loss


def train_baseline_rnn(
    config: Dict,
    Y,
    input_dim: int, 
    test_run: int = 0,
    checkpoint_dir=None,
):
    """
    Helper function for ray-tune to run hp search.
    """
    # Parse the configuration for current run
    epochs, lr, gamma = config['epochs'], config['lr'], config['gamma']

    if test_run > 0:
        # Try loading from checkpoint
        try:
            pl_model = BaselineRnn.load_from_checkpoint(os.path.join(os.getcwd(), f'results/rnn_{test_run}.ckpt'))
        except Exception:
            pl_model = None
    if test_run <= 0 or pl_model is None:
        rnn = RNNS[config['rnn_class']](
            input_size=input_dim,
            hidden_size=config['hidden_dim'],
        )
        pl_model = BaselineRnn(
            rnn=rnn,
            hidden_dim=config['hidden_dim'],
            output_dim=1,
            lr=lr, 
            gamma=gamma, 
        )
    
    data_module = ShiftingDataModule(
        Y=Y,
        seq_len=config['seq_len'],
        batch_size=config['batch_size'], 
    )
    metrics = {'loss': 'ptl/loss', 'valid_loss': 'ptl/valid_loss'}
    callbacks = [
        TuneReportCallback(metrics, on='validation_end'), 
        EarlyStopping(monitor='ptl/valid_loss'),
    ]

    # Run
    trainer = Trainer(
        default_root_dir=os.path.join(os.getcwd(), 'data'),
        max_epochs=epochs,
        gpus=1,
        logger=WandbLogger(
            project='mclatte-test', 
            log_model=True, 
            name='|'.join([
                f'{k} = {v:.3}' 
                if isinstance(v, float) 
                else f'{k} = {v}'
                for k, v in config.items()
            ])
        ),
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
    )
    
    trainer.fit(pl_model, data_module)
    if test_run > 0:
        trainer.save_checkpoint(os.path.join(os.getcwd(), f'results/rnn_{test_run}.ckpt'))
    return pl_model
