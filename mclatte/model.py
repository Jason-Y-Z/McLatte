"""
McLatte model definition and training procedure.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import abc
import os
import pytorch_lightning as pl
import torch
from mclatte.dataset import TimeSeriesDataModule
from mclatte.repr_learn import ENCODERS, DECODERS
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch import nn
from typing import Callable, Dict


class McEspresso(pl.LightningModule, abc.ABC):
    """
    Essence of the McLatte flavours, which defines 
    common parameters and configurations.
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lambda_r: float,  # reconstruction loss hp
        lambda_p: float,  # prognosis loss hp
        lr: float,
        gamma: float,  # learning rate decay
        post_trt_seq_len: int,
        hidden_dim: int,
    ):
        super().__init__()

        self._encoder = encoder.to(self.device)
        self._decoder = decoder.to(self.device)
        self._lambda_r = lambda_r
        self._lambda_p = lambda_p
        self._lr = lr
        self._gamma = gamma
        self.q_prog = nn.Parameter(torch.ones(post_trt_seq_len, hidden_dim) * 1.0e-4)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self._gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    @abc.abstractmethod
    def forward(self, x, a, t, mask):
        ...

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        ...

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        ...


class SkimmedMcLatte(McEspresso):
    """
    McLatte with no diagnostic loss and pre-treatment
    latent factor equal the post-treatment value.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

    def forward(self, x, a, t, mask):
        C = self._encoder(
            torch.transpose(x, 0, 1),
            a,
            torch.transpose(t, 0, 1),
            torch.transpose(mask, 0, 1),
        )
        x_tilde = torch.transpose(self._decoder(C), 0, 1)
        y_tilde = torch.matmul(C, self.q_prog.T)
        return x_tilde, y_tilde

    def training_step(self, batch, batch_idx):
        x, a, t, mask, _, y_post = batch

        x_tilde, y_tilde = self(x, a, t, mask)
        recon_loss = self._lambda_r * torch.mean(
            torch.linalg.matrix_norm((x - x_tilde) * mask, ord=2)
        )
        sp_loss = self._lambda_p * torch.nn.functional.mse_loss(y_tilde, y_post)
        loss = recon_loss + sp_loss

        self.log("ptl/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, a, t, mask, _, y_post = batch

        _, y_tilde = self(x, a, t, mask)
        loss = torch.nn.functional.l1_loss(y_tilde, y_post)

        self.log("ptl/valid_loss", loss)
        return loss


class SemiSkimmedMcLatte(McEspresso):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lambda_r: float,  # reconstruction loss hp
        lambda_d: float,  # diagnosis loss hp
        lambda_p: float,  # prognosis loss hp
        lr: float,
        gamma: float,  # learning rate decay
        pre_trt_seq_len: int,
        post_trt_seq_len: int,
        hidden_dim: int,
    ):
        super().__init__(
            encoder,
            decoder,
            lambda_r,
            lambda_p,
            lr,
            gamma,
            post_trt_seq_len,
            hidden_dim,
        )

        self.save_hyperparameters()
        self._lambda_d = lambda_d
        self.q_diag = nn.Parameter(torch.ones(pre_trt_seq_len, hidden_dim) * 1.0e-4)

    def forward(self, x, a, t, mask):
        C = self._encoder(
            torch.transpose(x, 0, 1),
            a,
            torch.transpose(t, 0, 1),
            torch.transpose(mask, 0, 1),
        )
        x_tilde = torch.transpose(self._decoder(C), 0, 1)
        y_diag = torch.matmul(C, self.q_diag.T)
        y_prog = torch.matmul(C, self.q_prog.T)
        return x_tilde, y_diag, y_prog

    def training_step(self, batch, batch_idx):
        x, a, t, mask, y_pre, y_post = batch

        x_tilde, y_diag, y_prog = self(x, a, t, mask)
        recon_loss = torch.mean(torch.linalg.matrix_norm((x - x_tilde) * mask, ord=2))
        diag_loss = torch.nn.functional.mse_loss(y_diag, y_pre)
        prog_loss = torch.nn.functional.mse_loss(y_prog, y_post)
        loss = (
            self._lambda_r * recon_loss
            + self._lambda_d * diag_loss
            + self._lambda_p * prog_loss
        )

        self.log("ptl/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, a, t, mask, _, y_post = batch

        _, _, y_tilde = self(x, a, t, mask)
        loss = torch.nn.functional.l1_loss(y_tilde, y_post)

        self.log("ptl/valid_loss", loss)
        return loss


class McLatte(SemiSkimmedMcLatte):
    def forward(self, x, a, t, mask):
        no_op = torch.zeros_like(a)
        C_pre = self._encoder(
            torch.transpose(x, 0, 1),
            no_op,
            torch.transpose(t, 0, 1),
            torch.transpose(mask, 0, 1),
        )
        C_post = self._encoder(
            torch.transpose(x, 0, 1),
            a,
            torch.transpose(t, 0, 1),
            torch.transpose(mask, 0, 1),
        )
        x_tilde = torch.transpose(self._decoder(C_pre), 0, 1)
        y_diag = torch.matmul(C_pre, self.q_diag.T)
        y_prog = torch.matmul(C_post, self.q_prog.T)
        return x_tilde, y_diag, y_prog


def train_mcespresso(
    config: Dict,
    X,
    M_,
    Y_pre,
    Y_post,
    A,
    T,
    R: int,
    M: int,
    input_dim: int,
    treatment_dim: int,
    ckpt_path: str,
    make_pl_model: Callable,
    test_run: int = 0,
):
    """
    Helper function for ray-tune to run hp search.
    """
    # Parse the configuration for current run
    epochs = config["epochs"]

    encoder = ENCODERS[config["encoder_class"]](
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        treatment_dim=treatment_dim,
    )
    decoder = DECODERS[config["decoder_class"]](
        hidden_dim=config["hidden_dim"],
        output_dim=input_dim,
        max_seq_len=R * M,
    )
    pl_model = make_pl_model(encoder, decoder)

    data_module = TimeSeriesDataModule(
        X=X,
        M=M_,
        Y_pre=Y_pre,
        Y_post=Y_post,
        A=A,
        T=T,
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


def train_skimmed_mclatte(
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
    lr, gamma = config["lr"], config["gamma"]
    ckpt_path = os.path.join(os.getcwd(), f"results/skimmed_mclatte_{test_run}.ckpt")
    if test_run > 0:
        # Try loading from checkpoint
        try:
            return SkimmedMcLatte.load_from_checkpoint(ckpt_path)
        except Exception as e:
            print(e)

    def make_skimmed_mclatte(encoder, decoder):
        return SkimmedMcLatte(
            encoder=encoder,
            decoder=decoder,
            lambda_r=config["lambda_r"],
            lambda_p=config["lambda_p"],
            lr=lr,
            gamma=gamma,
            post_trt_seq_len=H,
            hidden_dim=config["hidden_dim"],
        )

    return train_mcespresso(
        config,
        X,
        M_,
        Y_pre,
        Y_post,
        A,
        T,
        R,
        M,
        input_dim,
        treatment_dim,
        ckpt_path,
        make_skimmed_mclatte,
        test_run,
    )


def train_semi_skimmed_mclatte(
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
    lr, gamma = config["lr"], config["gamma"]
    ckpt_path = os.path.join(
        os.getcwd(), f"results/semi_skimmed_mclatte_{test_run}.ckpt"
    )
    if test_run > 0:
        # Try loading from checkpoint
        try:
            return SemiSkimmedMcLatte.load_from_checkpoint(ckpt_path)
        except Exception as e:
            print(e)

    def make_semi_skimmed_mclatte(encoder, decoder):
        return SemiSkimmedMcLatte(
            encoder=encoder,
            decoder=decoder,
            lambda_r=config["lambda_r"],
            lambda_d=config["lambda_d"],
            lambda_p=config["lambda_p"],
            lr=lr,
            gamma=gamma,
            pre_trt_seq_len=M,
            post_trt_seq_len=H,
            hidden_dim=config["hidden_dim"],
        )

    return train_mcespresso(
        config,
        X,
        M_,
        Y_pre,
        Y_post,
        A,
        T,
        R,
        M,
        input_dim,
        treatment_dim,
        ckpt_path,
        make_semi_skimmed_mclatte,
        test_run,
    )


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
    lr, gamma = config["lr"], config["gamma"]
    ckpt_path = os.path.join(os.getcwd(), f"results/mclatte_{test_run}.ckpt")
    if test_run > 0:
        # Try loading from checkpoint
        try:
            return McLatte.load_from_checkpoint(ckpt_path)
        except Exception as e:
            print(e)

    def make_mclatte(encoder, decoder):
        return McLatte(
            encoder=encoder,
            decoder=decoder,
            lambda_r=config["lambda_r"],
            lambda_d=config["lambda_d"],
            lambda_p=config["lambda_p"],
            lr=lr,
            gamma=gamma,
            pre_trt_seq_len=M,
            post_trt_seq_len=H,
            hidden_dim=config["hidden_dim"],
        )

    return train_mcespresso(
        config,
        X,
        M_,
        Y_pre,
        Y_post,
        A,
        T,
        R,
        M,
        input_dim,
        treatment_dim,
        ckpt_path,
        make_mclatte,
        test_run,
    )
