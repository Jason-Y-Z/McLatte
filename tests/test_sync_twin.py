import numpy as np
import os
import pytest
import torch
from mclatte.simulation_data import TreatmentRepr, generate_simulation_data
from pytorch_lightning import Trainer
from synctwin.dataset import SyncTwinDataModule
from synctwin.decoder import RegularDecoder
from synctwin.encoder import RegularEncoder
from synctwin.model import SyncTwin, SyncTwinPl


# Given
@pytest.mark.parametrize('N', (20, ))
@pytest.mark.parametrize('M', (5, ))
@pytest.mark.parametrize('H', (5, ))
@pytest.mark.parametrize('R', (5, ))
@pytest.mark.parametrize('D', (10, ))
@pytest.mark.parametrize('K', (2, ))
@pytest.mark.parametrize('C', (5, ))
@pytest.mark.parametrize(
    'mode', 
    (
        TreatmentRepr.BINARY, 
        # TreatmentRepr.BOUNDED, 
        # TreatmentRepr.REAL_VALUED
    )
)
def test_sync_twin(N, M, H, R, D, K, C, mode):
    X, M_, _, Y_post, A, T = generate_simulation_data(N, M, H, R, D, K, C, mode)
    Y_mask = np.all(A == 0, axis=1)
    Y_control = Y_post[Y_mask]
    enc = RegularEncoder(input_dim=D, hidden_dim=5)
    dec = RegularDecoder(hidden_dim=enc.hidden_dim, output_dim=enc.input_dim, max_seq_len=R * M)
    sync_twin = SyncTwin(
        n_unit=Y_control.shape[0],
        n_treated=N - Y_control.shape[0],
        reg_B=0,
        lam_express=1,
        lam_recon=1,
        lam_prognostic=1,
        tau=1,
        encoder=enc,
        decoder=dec,
    )
    data_module = SyncTwinDataModule(
        X=X,
        M=M_,
        T=T,
        Y_batch=Y_post,
        Y_mask=Y_mask,
        batch_size=3, 
    )

    # Run
    trainer = Trainer(
        default_root_dir=os.path.join(os.getcwd(), 'tests/data'),
        max_epochs=1,
        progress_bar_refresh_rate=0,
        log_every_n_steps=1,
        gpus=1,
    )
    pl_model = SyncTwinPl(
        sync_twin=sync_twin, 
        lr=1e-2, 
        gamma=0.9, 
        y_control=torch.from_numpy(Y_control).float().cuda(),
    )
    trainer.fit(pl_model, data_module)
