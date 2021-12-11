import os
import pytest
from mclatte.dataset import TimeSeriesDataModule
from mclatte.model import McLatte
from mclatte.repr_learn import ENCODERS, DECODERS
from mclatte.simulation_data import generate_simulation_data, TreatmentRepr
from pytorch_lightning import Trainer


# Given
@pytest.mark.parametrize('N', (5, ))
@pytest.mark.parametrize('M', (5, ))
@pytest.mark.parametrize('H', (5, ))
@pytest.mark.parametrize('R', (5, ))
@pytest.mark.parametrize('D', (10, ))
@pytest.mark.parametrize('K', (5, ))
@pytest.mark.parametrize('C', (5, ))
@pytest.mark.parametrize(
    'mode', 
    (
        TreatmentRepr.BINARY, 
        TreatmentRepr.BOUNDED, 
        TreatmentRepr.REAL_VALUED
    )
)
@pytest.mark.parametrize('encoder', ('lstm', ))
@pytest.mark.parametrize('decoder', ('lstm', ))
def test_mclatte(N, M, H, R, D, K, C, mode, encoder, decoder):
    X, M_, Y_pre, Y_post, A, T = generate_simulation_data(N, M, H, R, D, K, C, mode)
    data_module = TimeSeriesDataModule(
        X=X,
        M=M_,
        Y_pre=Y_pre,
        Y_post=Y_post,
        A=A, 
        T=T,
        batch_size=16,
    )
    mclatte = McLatte(
        encoder=ENCODERS[encoder](input_dim=D, hidden_dim=C, treatment_dim=K), 
        decoder=DECODERS[decoder](hidden_dim=C, output_dim=D, max_seq_len=R * M), 
        lambda_r=1, 
        lambda_s=1, 
        lr=1e-2, 
        gamma=0.9, 
        post_trt_seq_len=H, 
        hidden_dim=C,
    )
    trainer = Trainer(
        default_root_dir=os.path.join(os.getcwd(), 'tests/data'),
        max_epochs=1,
        progress_bar_refresh_rate=0,
        log_every_n_steps=1,
        gpus=1,
    )
    trainer.fit(mclatte, data_module)
