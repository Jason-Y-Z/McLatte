""" 
SyncTwin model, adapted from
https://github.com/vanderschaarlab/SyncTwin-NeurIPS-2021
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from synctwin._config import D_TYPE, DEVICE
from synctwin.dataset import SyncTwinDataModule
from synctwin.decoder import RegularDecoder
from synctwin.encoder import RegularEncoder
from typing import Dict


class SyncTwin(nn.Module):
    def __init__(
        self,
        n_unit,
        n_treated,
        reg_B=0.0,
        lam_express=1.0,
        lam_recon=0.0,
        lam_prognostic=0.0,
        tau=1.0,
        encoder=None,
        decoder=None,
        decoder_Y=None,
        dtype=D_TYPE,
        device=DEVICE,
        reduce_gpu_memory=False,
        inference_only=False,
    ):
        super(SyncTwin, self).__init__()
        assert not (reduce_gpu_memory and inference_only)

        self.n_unit = n_unit
        self.n_treated = n_treated
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        if decoder_Y is not None:
            self.decoder_Y = decoder_Y
        if reduce_gpu_memory:
            init_B = (torch.ones(1, 1, dtype=dtype, device=device)) * 1.0e-4
        elif inference_only:
            init_B = (
                torch.ones(n_treated, n_unit, dtype=dtype, device=device)
            ) * 1.0e-4
        else:
            init_B = (
                torch.ones(n_unit + n_treated, n_unit, dtype=dtype, device=device)
            ) * 1.0e-4
        self.B = nn.Parameter(init_B)
        self.C0 = torch.zeros(
            n_unit,
            self.encoder.hidden_dim,
            dtype=dtype,
            requires_grad=False,
            device=device,
        )
        # regularization strength of matrix B
        self.reg_B = reg_B

        self.lam_recon = lam_recon
        self.lam_prognostic = lam_prognostic
        self.lam_express = lam_express
        self.tau = tau

    def get_representation(self, x, t, mask):
        # get representation C: B(atch size), D(im hidden)
        C = self.encoder(x, t, mask)
        return C

    def get_reconstruction(self, C, t, mask):
        x_hat = self.decoder(C, t, mask)
        return x_hat

    def get_prognostics(self, C, t, mask):
        y_hat = self.decoder_Y(C, t, mask)
        return y_hat

    def get_B_reduced(self, batch_ind):
        # B * N0
        batch_index = torch.stack([batch_ind] * self.n_unit, dim=-1)
        B_reduced = torch.gather(self.B, 0, batch_index)
        B_reduced = F.gumbel_softmax(B_reduced, tau=self.tau, dim=1, hard=False)
        return B_reduced

    def update_C0(self, C, batch_ind):
        # in total data matrix, control first, treated second
        for i, ib in enumerate(batch_ind):
            if ib < self.n_unit:
                self.C0[ib] = C[i].detach()

    def self_expressive_loss(self, C, B_reduced):

        err = C - torch.matmul(B_reduced, self.C0)
        err_mse = torch.mean(err[~torch.isnan(err)] ** 2)

        # L2 regularization
        reg = torch.mean(B_reduced[~torch.isnan(B_reduced)] ** 2)
        return self.lam_express * (err_mse + self.reg_B * reg)

    def reconstruction_loss(self, x, x_hat, mask):
        if self.lam_recon == 0:
            return 0
        err = (x - x_hat) * mask
        err_mse = torch.sum(err ** 2) / torch.sum(mask)
        return err_mse * self.lam_recon

    def prognostic_loss(self, B_reduced, y_batch, y_control, y_mask):
        # y_batch: B, DY
        # y_mask: B (1 if control, 0 if treated)
        # y_all: N0, DY
        # B_reduced: B, N0
        y_hat = torch.matmul(B_reduced, y_control)
        mse = (y_batch - y_hat) ** 2
        masked_mse = torch.sum(mse * y_mask.unsqueeze(-1))
        return (
            masked_mse / torch.sum(y_mask) * self.lam_prognostic,
            torch.nn.functional.l1_loss(y_hat, y_batch),
        )

    def prognostic_loss2(self, y, y_hat, mask):
        err = (y - y_hat) * mask
        err_mse = torch.sum(err ** 2) / torch.sum(mask)
        return err_mse * self.lam_prognostic

    def forward(self, x, t, mask, batch_ind, y_batch, y_control, y_mask):
        C = self.get_representation(x, t, mask)
        x_hat = self.get_reconstruction(C, t, mask)

        B_reduced = self.get_B_reduced(batch_ind)
        self_expressive_loss = self.self_expressive_loss(C, B_reduced)
        reconstruction_loss = self.reconstruction_loss(x, x_hat, mask)
        prognostic_loss, l1_loss = self.prognostic_loss(
            B_reduced, y_batch, y_control, y_mask
        )
        return self_expressive_loss + reconstruction_loss + prognostic_loss, l1_loss, C


class SyncTwinPl(pl.LightningModule):
    def __init__(
        self,
        sync_twin: SyncTwin,
        lr: float,
        gamma: float,  # learning rate decay
        y_control: torch.Tensor,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        self._sync_twin = sync_twin
        self._lr = lr
        self._gamma = gamma
        self._y_control = y_control

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self._gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def forward(self, x, t, mask, batch_ind, y_batch, y_mask):
        loss, l1_loss, _ = self._sync_twin(
            torch.transpose(x, 0, 1),
            torch.transpose(t, 0, 1),
            torch.transpose(mask, 0, 1),
            batch_ind,
            y_batch,
            self._y_control,
            y_mask,
        )
        return loss, l1_loss

    def training_step(self, batch, batch_idx):
        x, t, mask, batch_ind, y_batch, y_mask = batch

        loss, _ = self(x, t, mask, batch_ind, y_batch, y_mask)

        self.log("ptl/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, mask, batch_ind, y_batch, y_mask = batch

        _, loss = self(x, t, mask, batch_ind, y_batch, y_mask)

        self.log("ptl/valid_loss", loss)
        return loss


def train_synctwin(
    config: Dict,
    X,
    M_,
    T,
    Y_batch,
    Y_control,
    Y_mask,
    N: int,
    D: int,
    n_treated: int,
    pre_trt_x_len: int,
    test_run: int = 0,
    device=DEVICE,
    ckpt_path=None,
    checkpoint_dir=None,  # kept for compatibility with ray[tune]
):
    """
    Helper function for ray-tune to run hp search.
    """
    # Parse the configuration for current run
    epochs, lr, gamma = config["epochs"], config["lr"], config["gamma"]
    ckpt_path = (
        ckpt_path
        if ckpt_path is not None
        else os.path.join(os.getcwd(), f"results/synctwin_{test_run}.ckpt")
    )

    if test_run > 0:
        # Try loading from checkpoint
        try:
            return SyncTwinPl.load_from_checkpoint(ckpt_path)
        except Exception as e:
            print(e)

    enc = RegularEncoder(input_dim=D, hidden_dim=config["hidden_dim"])
    dec = RegularDecoder(
        hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=pre_trt_x_len
    )
    sync_twin = SyncTwin(
        n_unit=N - n_treated,
        n_treated=n_treated,
        reg_B=config["reg_B"],
        lam_express=config["lam_express"],
        lam_recon=config["lam_recon"],
        lam_prognostic=config["lam_prognostic"],
        tau=config["tau"],
        encoder=enc,
        decoder=dec,
        device=device,
    )
    pl_model = SyncTwinPl(
        sync_twin=sync_twin,
        lr=lr,
        gamma=gamma,
        y_control=torch.from_numpy(Y_control).float().to(device),
    )

    data_module = SyncTwinDataModule(
        X=X,
        M=M_,
        T=T,
        Y_batch=Y_batch,
        Y_mask=Y_mask,
        batch_size=config["batch_size"],
    )
    metrics = {"loss": "ptl/loss", "valid_loss": "ptl/valid_loss"}
    callbacks = (
        [TuneReportCallback(metrics, on="validation_end")] if test_run == 0 else []
    )
    callbacks.append(EarlyStopping(monitor="ptl/valid_loss"))
    logger = (
        False
        if test_run > 0
        else WandbLogger(
            project="mclatte-test",
            log_model=True,
            name="|".join(
                [
                    f"{k} = {v:.3}" if isinstance(v, float) else f"{k} = {v}"
                    for k, v in config.items()
                ]
            ),
        )
    )

    # Run
    trainer = Trainer(
        default_root_dir=os.path.join(os.getcwd(), "data"),
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
        devices=4,
        accelerator="auto",
    )
    trainer.fit(pl_model, data_module)
    if test_run > 0:
        trainer.save_checkpoint(ckpt_path)

    return pl_model
