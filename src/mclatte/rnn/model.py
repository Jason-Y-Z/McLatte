"""
Baseline RNN model for time series predictions.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import os
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from torch import nn

from mclatte.rnn.dataset import ShiftingDataModule


RNNS = {
    "rnn": torch.nn.RNN,
    "lstm": torch.nn.LSTM,
    "gru": torch.nn.GRU,
}


class BaselineRnn(pl.LightningModule):
    """
    Baseline recurent neural network.
    """

    def __init__(
        self,
        rnn: nn.Module,
        hidden_dim: int,
        output_dim: int,
        lr: float,
        gamma: float,
    ):
        super().__init__()

        self.save_hyperparameters()
        self._rnn = rnn.to(self.device)
        self._linear = nn.Linear(hidden_dim, output_dim)
        self._lr = lr
        self._gamma = gamma

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self._gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def forward(self, y):  # pylint: disable=arguments-differ
        output, _ = self._rnn(torch.transpose(y, 0, 1))
        y_tilde = torch.transpose(self._linear(output), 0, 1)
        return y_tilde

    def training_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ, unused-argument
        y, y_shifted = batch

        y_tilde = self(y)
        loss = torch.nn.functional.mse_loss(y_tilde, y_shifted)

        self.log("ptl/loss", loss)
        return loss

    def validation_step(
        self, batch, batch_idx
    ):  # pylint: disable=arguments-differ, unused-argument
        y, y_shifted = batch

        y_tilde = self(y)
        loss = torch.nn.functional.l1_loss(y_tilde, y_shifted)

        self.log("ptl/valid_loss", loss)
        return loss


def train_baseline_rnn(
    config: Dict,
    Y,
    input_dim: int,
    test_run: int = 0,
    checkpoint_dir=None,  # pylint: disable=unused-argument
):
    """
    Helper function for ray-tune to run hp search.
    """
    # Parse the configuration for current run
    epochs, lr, gamma = config["epochs"], config["lr"], config["gamma"]

    if test_run > 0:
        # Try loading from checkpoint
        try:
            return BaselineRnn.load_from_checkpoint(
                os.path.join(os.getcwd(), f"results/rnn_{test_run}.ckpt")
            )
        except Exception as e:  # pylint: disable=broad-except
            print(e)

    rnn = RNNS[config["rnn_class"]](
        input_size=input_dim,
        hidden_size=config["hidden_dim"],
    )
    pl_model = BaselineRnn(
        rnn=rnn,
        hidden_dim=config["hidden_dim"],
        output_dim=1,
        lr=lr,
        gamma=gamma,
    )

    data_module = ShiftingDataModule(
        Y=Y,
        seq_len=config["seq_len"],
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
    )

    trainer.fit(pl_model, data_module)
    if test_run > 0:
        trainer.save_checkpoint(
            os.path.join(os.getcwd(), f"results/rnn_{test_run}.ckpt")
        )
    return pl_model


def rnn_predict(trained_rnn, Y_pre, Y_post, return_Y_pred=False):
    """
    Make predictions using results from previous time steps.
    """
    Y = Y_pre
    losses = 0.0
    Y_pred = []
    for i in range(Y_post.shape[1]):
        Y_tilde = trained_rnn(torch.from_numpy(Y).float().unsqueeze(2)).squeeze()

        Y = np.concatenate((Y[:, 1:], Y_tilde.cpu().detach().numpy()[:, [-1]]), axis=1)

        losses += torch.nn.functional.l1_loss(
            Y_tilde[:, -1], torch.from_numpy(Y_post).float()[:, i]
        ).item()
        Y_pred.append(Y_tilde[:, -1])
    if return_Y_pred:
        return torch.stack(Y_pred, 1)
    return losses / Y_post.shape[1]


def infer_rnn(trained_rnn, Y_pre_test, Y_post_test, return_Y_pred=False):
    trained_rnn.eval()
    return rnn_predict(trained_rnn, Y_pre_test, Y_post_test, return_Y_pred)
