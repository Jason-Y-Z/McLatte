import os
import numpy as np
import pytest
import torch
from mclatte.mclatte.simulation_data import generate_simulation_data, TreatmentRepr
from mclatte.rnn.model import BaselineRnn
from mclatte.rnn.dataset import ShiftingDataModule
from pytorch_lightning import Trainer


# Given
@pytest.mark.parametrize("N", (20,))
@pytest.mark.parametrize("M", (100,))
@pytest.mark.parametrize("H", (100,))
@pytest.mark.parametrize("R", (1,))
@pytest.mark.parametrize("D", (10,))
@pytest.mark.parametrize("K", (5,))
@pytest.mark.parametrize("C", (5,))
@pytest.mark.parametrize("mode", (TreatmentRepr.BINARY,))
def test_baseline_rnn(N, M, H, R, D, K, C, mode):
    _, _, Y_pre, Y_post, _, _ = generate_simulation_data(N, M, H, R, D, K, C, mode)
    data_module = ShiftingDataModule(
        Y=np.concatenate((Y_pre, Y_post), axis=1),
        seq_len=16,
        batch_size=8,
    )
    rnn = BaselineRnn(
        rnn=torch.nn.RNN(input_size=1, hidden_size=4),
        hidden_dim=4,
        output_dim=1,
        lr=1e-2,
        gamma=0.9,
    )
    trainer = Trainer(
        default_root_dir="tests/data/baseline_rnn",
        max_epochs=1,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(rnn, data_module)
