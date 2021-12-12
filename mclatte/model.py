""" 
McLatte model definition and training procedure.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from mclatte.dataset import TimeSeriesDataModule
from mclatte.repr_learn import ENCODERS, DECODERS
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from typing import Dict


class McLatte(pl.LightningModule):
    def __init__(
        self, 
        encoder: nn.Module, 
        decoder: nn.Module, 
        lambda_r: float,  # reconstruction loss hp
        lambda_s: float,  # supervised loss hp
        lr: float,
        gamma: float,  # learning rate decay
        post_trt_seq_len: int, 
        hidden_dim: int, 
    ):
        super().__init__()

        self.save_hyperparameters()
        self._encoder = encoder.to(self.device)
        self._decoder = decoder.to(self.device)
        self._lambda_r = lambda_r
        self._lambda_s = lambda_s
        self._lr = lr
        self._gamma = gamma
        self.q_tilde = nn.Parameter(torch.ones(post_trt_seq_len, hidden_dim) * 1.0e-4)
    
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

    def forward(self, x, a, t, mask):
        C = self._encoder(
            torch.transpose(x, 0, 1), 
            a, 
            torch.transpose(t, 0, 1), 
            torch.transpose(mask, 0, 1),
        )
        x_tilde = torch.transpose(self._decoder(C), 0, 1)
        y_tilde = torch.matmul(C, self.q_tilde.T) 
        return x_tilde, y_tilde
    
    def training_step(self, batch, batch_idx):
        x, a, t, mask, _, y_post = batch

        x_tilde, y_tilde = self(x, a, t, mask)
        recon_loss = self._lambda_r * torch.mean(torch.linalg.matrix_norm((x - x_tilde) * mask, ord=2))
        sp_loss = self._lambda_s * torch.nn.functional.mse_loss(y_tilde, y_post)
        loss = recon_loss + sp_loss
        
        self.log('ptl/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, a, t, mask, _, y_post = batch

        _, y_tilde = self(x, a, t, mask)
        loss = torch.nn.functional.l1_loss(y_tilde, y_post)

        self.log('ptl/valid_loss', loss)
        return loss


def train_mclatte(
    config: Dict,
    X,
    M_,
    Y_pre,
    Y_post,
    A, 
    T,
    R: int,
    M: int,
    H: int,
    input_dim: int, 
    treatment_dim: int, 
    test_run: int = 0,
    checkpoint_dir=None,  # kept for compatibility with ray[tune]
):
    """
    Helper function for ray-tune to run hp search.
    """
    # Parse the configuration for current run
    epochs, lr, gamma = config['epochs'], config['lr'], config['gamma']

    if test_run > 0:
        # Try loading from checkpoint
        try:
            pl_model = McLatte.load_from_checkpoint(os.path.join(os.getcwd(), f'results/mclatte_{test_run}.ckpt'))
        except Exception as e:
            print(e)
            pl_model = None

    if test_run <= 0 or pl_model is None: 
        encoder = ENCODERS[config['encoder_class']](
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            treatment_dim=treatment_dim,
        )
        decoder = DECODERS[config['decoder_class']](
            hidden_dim=config['hidden_dim'], 
            output_dim=input_dim, 
            max_seq_len=R * M,
        )
        pl_model = McLatte(
            encoder=encoder, 
            decoder=decoder, 
            lambda_r=config['lambda_r'], 
            lambda_s=config['lambda_s'], 
            lr=lr, 
            gamma=gamma, 
            post_trt_seq_len=H, 
            hidden_dim=config['hidden_dim']
        )

    data_module = TimeSeriesDataModule(
        X=X,
        M=M_,
        Y_pre=Y_pre,
        Y_post=Y_post,
        A=A, 
        T=T,
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
        trainer.save_checkpoint(os.path.join(os.getcwd(), f'results/mclatte_{test_run}.ckpt'))
    return pl_model
