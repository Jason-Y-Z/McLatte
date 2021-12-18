import pytest
from mclatte.test_data.idt import (
    generate_simulation_data,
    simulate_covariates,
    simulate_masking_vectors,
    simulate_outcomes,
    simulate_treatment_vectors,
    TreatmentRepr,
)


# Given
@pytest.mark.parametrize("N", (50,))
@pytest.mark.parametrize("L", (5,))
@pytest.mark.parametrize("P", (5,))
@pytest.mark.parametrize("T", (50,))
def test_simulate_covariates(N, L, P, T):
    # When
    X = simulate_covariates(N, L, P, T)

    # Then
    assert X.shape == (N, T, L + P)


# Given
@pytest.mark.parametrize("N", (50,))
@pytest.mark.parametrize("D", (10,))
@pytest.mark.parametrize("T", (50,))
def test_simulate_masking_vectors(N, D, T):
    # When
    M = simulate_masking_vectors(N, D, T)

    # Then
    assert M.shape == (N, T, D)
    assert set(list(M.reshape((-1)))) <= {0, 1}


# Given
@pytest.mark.parametrize("N", (50,))
@pytest.mark.parametrize("K", (10,))
@pytest.mark.parametrize(
    "mode", (TreatmentRepr.BINARY, TreatmentRepr.BOUNDED, TreatmentRepr.REAL_VALUED)
)
def test_simulate_treatment_vectors(N, K, mode):
    # When
    A = simulate_treatment_vectors(N, K, mode)

    # Then
    assert A.shape == (N, K)


# Given
@pytest.mark.parametrize("N", (5,))
@pytest.mark.parametrize("M", (5,))
@pytest.mark.parametrize("H", (5,))
@pytest.mark.parametrize("L", (5,))
@pytest.mark.parametrize("P", (5,))
@pytest.mark.parametrize("R", (5,))
@pytest.mark.parametrize("K", (10,))
@pytest.mark.parametrize("C", (5,))
@pytest.mark.parametrize(
    "mode", (TreatmentRepr.BINARY, TreatmentRepr.BOUNDED, TreatmentRepr.REAL_VALUED)
)
def test_simulate_outcomes(N, M, H, L, P, R, K, C, mode):
    D = L + P
    T = M + H
    X = simulate_covariates(N, L, P, T * R)
    A = simulate_treatment_vectors(N, K, mode)

    # When
    Y_pre, Y_post = simulate_outcomes(X, A, N, M, H, R, D, K, C)

    # Then
    assert Y_pre.shape == (N, M)
    assert Y_post.shape == (N, H)


# Given
@pytest.mark.parametrize("N", (5,))
@pytest.mark.parametrize("M", (5,))
@pytest.mark.parametrize("H", (5,))
@pytest.mark.parametrize("R", (5,))
@pytest.mark.parametrize("D", (10,))
@pytest.mark.parametrize("K", (5,))
@pytest.mark.parametrize("C", (5,))
@pytest.mark.parametrize(
    "mode", (TreatmentRepr.BINARY, TreatmentRepr.BOUNDED, TreatmentRepr.REAL_VALUED)
)
def test_generate_simulation_data(N, M, H, R, D, K, C, mode):
    # When
    X, M_, Y_pre, Y_post, A, T = generate_simulation_data(N, M, H, R, D, K, C, mode)

    # Then
    assert X.shape == (N, R * M, D)
    assert M_.shape == (N, R * M, D)
    assert Y_pre.shape == (N, M)
    assert Y_post.shape == (N, H)
    assert A.shape == (N, K)
    assert T.shape == (N, R * M, D)
