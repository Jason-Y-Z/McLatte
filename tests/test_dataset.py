import pytest
from mclatte.dataset import TimeSeriesDataset
from mclatte.simulation_data import generate_simulation_data, TreatmentRepr
from torch.utils.data import DataLoader


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
def test_ts_dataset(N, M, H, R, D, K, C, mode):
    X, M_, Y_pre, Y_post, A, T = generate_simulation_data(N, M, H, R, D, K, C, mode)
    batch_size = 4

    # When
    dataset = TimeSeriesDataset(
        X=X,
        M=M_,
        Y_pre=Y_pre,
        Y_post=Y_post,
        A=A, 
        T=T,
    )
    x, a, t, mask, y_pre, y_post = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=True)))

    # Then
    assert x.shape == (batch_size, R * M, D)
    assert a.shape == (batch_size, K)
    assert t.shape == (batch_size, R * M, D)
    assert mask.shape == (batch_size, R * M, D)
    assert y_pre.shape == (batch_size, M)
    assert y_post.shape == (batch_size, H)
