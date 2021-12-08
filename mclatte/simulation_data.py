""" 
Data generation utilities for the simulation study.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import enum
import numpy as np
from sklearn.decomposition import PCA
from typing import Callable, Tuple


def _sample_for_N_T(
    N: int, 
    T: int, 
    sample: Callable
) -> np.ndarray:
    """
    Draw [N, T] samples using the sample function given.

    Parameters
    ----------
    N : int
        First dimension of the result.
    
    T : int
        Second dimension of the result.

    sample : Callable[[int, int], np.ndarray of shape (D)]

    Returns
    -------
    X : np.ndarray of shape (N, T, D)
        Repetitively drawn samples using sample function.
    """
    X = []
    for i in range(N):
        X_i = [sample(i, t) for t in range(T)]
        X.append(np.stack(X_i))
    return np.stack(X)


def _simulate_multivariate_time_series(
    N: int,
    L: int,
    P: int,
    T: int, 
) -> np.ndarray:
    """
    Simulate a multivariate normal time series.

    Parameters
    ----------
    N : int
        Number of subjects in the series.

    L : int
        Number of time-linear features in the variables.

    P : int
        Number of periodic features in the variables.
    
    T : int
        Length of the series.

    Returns
    -------
    X : np.ndarray of shape (N, T, L + P)
        A multivariate normal time series, 
        constructed from a linear component and a periodic component.
    """
    t = np.arange(T)

    # Initialise parameters for the linear component
    alpha_linear = np.random.uniform(-1 / T, 1 / T, (N, L))
    beta_linear = np.random.uniform(-1, 1, (N, L))
    mu_linear = (
        alpha_linear[:, None, :] 
            * t[None, :, None] 
        + beta_linear[:, None, :]
    )
    sigma_linear = np.einsum(
        'ijk,ilk->ijl', 
        (comp_max := np.random.uniform(-1, 1, (N, L, L))), 
        comp_max
    )

    # Initialise parameters for the periodic component
    alpha_periodic = np.random.uniform(-1, 1, (N, P))
    beta_periodic = np.random.uniform(-1, 1, (N, P))
    omega_periodic = np.random.uniform(-np.pi, np.pi, (N, P))
    mu_periodic = (
        alpha_periodic[:, None, :] 
            * np.sin(omega_periodic[:, None, :] * t[None, :, None]) 
        + beta_periodic[:, None, :]
    )
    sigma_periodic = np.einsum(
        'ijk,ilk->ijl', 
        (comp_max := np.random.uniform(-1, 1, (N, P, P))), 
        comp_max
    )

    # Draw samples from the distributions defined by the initialised parameters
    def sample_x(i: int, t: int) -> np.ndarray:
        x_linear = np.random.multivariate_normal(
            mu_linear[i, t], 
            sigma_linear[i]
        )
        x_periodic = np.random.multivariate_normal(
            mu_periodic[i, t], 
            sigma_periodic[i]
        )
        x_it = np.concatenate([x_linear, x_periodic])
        return x_it
    
    return _sample_for_N_T(N, T, sample_x)


def simulate_covariates(
    N: int,
    L: int,
    P: int,
    T: int, 
) -> np.ndarray:
    """
    Simulate the covariates time series.

    Parameters
    ----------
    N : int
        Number of subjects in the study.

    L : int
        Number of time-linear features in the covariates.

    P : int
        Number of periodic features in the covariates.
    
    T : int
        Timespan of the simulation.

    Returns
    -------
    X : np.ndarray of shape (N, T, L + P)
        Covariates, constructed from a linear component and a periodic component.
    """
    return _simulate_multivariate_time_series(N, L, P, T)


def simulate_masking_vectors(
    N: int,
    D: int,
    T: int,
):
    """
    Simulate the masking vectors for the covariates.

    Parameters
    ----------
    N : int
        Number of subjects in the study.

    D : int
        Number of features in the covariates.
    
    T : int
        Timespan of the simulation.

    Returns
    -------
    M : np.ndarray of shape (N, T, D)
        Masking vectors, which indicate whether a certain feature in covariates is measured.
    """
    # Initialise parameters for the Bernoulli distribution
    p = np.random.rand(N, D)

    # Draw samples from the distributions defined by the initialised parameters
    return _sample_for_N_T(N, T, lambda i, _: np.random.binomial(np.ones(D, dtype=np.int32), p[i]))


class TreatmentRepr(enum.Enum):
    BINARY = 0
    BOUNDED = 1
    REAL_VALUED = 2


def simulate_treatment_vectors(
    N: int,
    K: int,
    mode: TreatmentRepr
):
    """
    Simulate the masking vectors for the covariates.

    Parameters
    ----------
    N : int
        Number of subjects in the study.

    K : int
        Number of causes in the treatment.
    
    mode : TreatmentRepr
        Representation of the treatment, can be binary, bounded or real-valued.

    Returns
    -------
    A : np.ndarray of shape (N, K)
        Treatment indicator vectors, which indicate the cause configurations for each subject.
    """
    if mode is TreatmentRepr.BINARY:
        return np.stack([
            np.random.binomial(
                np.ones(K, dtype=np.int32), 
                np.ones(K) / 2
            ) for _ in range(N)
        ])
    if mode is TreatmentRepr.BOUNDED:
        return np.random.rand(N, K)
    if mode is TreatmentRepr.REAL_VALUED:
        return np.random.normal(0, 1, (N, K))
    raise ValueError(f'Unrecognized treatment representation: mode = {mode}')


def simulate_outcomes(
    X: np.ndarray, 
    A: np.ndarray,
    N: int, 
    M: int,
    H: int, 
    R: int,  
    D: int,
    K: int,
    C: int,
) -> np.ndarray:
    """
    Simulate outcome measures for the study.

    Parameters
    ----------
    X : np.ndarray of shape (N, T * R, D)
        Covariates, constructed from a linear component and a periodic component.

    A : np.ndarray of shape (N, K)
        Treatment indicator vectors, which indicate the cause configurations for each subject.
        
    N : int
        Number of subjects in the study.

    M : int
        Pre-treatment timespan for outcome measurements (Y).
    
    H : int
        (Post-)treatment timespan for outcome measurements (Y).

    R : int
        Covariate-outcome measurement interval ratio.

    D : int
        Number of features in the covariates.

    K : int
        Number of causes in the treatment.
    
    C : int
        Dimension of the latent factors.

    Returns
    -------
    Y : np.ndarray of shape (N, T)
        Outcome measures, which span the pre-treatment and treatment time.
    """
    X_pre, X_post = X[:, :M * R, :], X[:, M * R:, :]
    
    g = np.random.rand(1, M * R)
    G = np.random.rand(K, H * R)
    H_ = np.random.rand(D, C)
    
    C_pre = np.stack([np.squeeze(
        g @ X_pre[i] @ H_
    ) for i in range(N)])
    C_post = np.stack([np.squeeze(
        A[i] @ G @ X_post[i] @ H_
    ) for i in range(N)])
    Q = _simulate_multivariate_time_series(
        1, 
        C // 3, 
        C - C // 3, 
        M + H
    ).reshape((M + H, C))
    
    Y_pre = _sample_for_N_T(
        N, 
        M, 
        lambda i, t: np.random.normal(np.dot(Q[t], C_pre[i]), 1)
    ).reshape((N, M))
    Y_post = _sample_for_N_T(
        N, 
        H, 
        lambda i, t: np.random.normal(np.dot(Q[t], C_post[i]), 1)
    ).reshape((N, H))
    return np.concatenate((Y_pre, Y_post), axis=1)


def generate_simulation_data(
    N: int,
    M: int,  
    H: int, 
    R: int,  
    D: int,
    K: int,
    C: int,
    treatment_repr: TreatmentRepr,
) -> Tuple[
    np.ndarray, 
    np.ndarray, 
    np.ndarray, 
    np.ndarray
]:
    """
    Generate data for simulation study.

    Parameters
    ----------
    N : int
        Number of subjects in the study.

    M : int
        Pre-treatment timespan for outcome measurements (Y).
    
    H : int
        (Post-)treatment timespan for outcome measurements (Y).
    
    R : int
        Covariate measurements happen on a different time granularity from the outcome,
        for example in the time interval between two outcome measurements, we can make R
        covariate measurements.

    D : int
        Number of features in the covariates.
    
    K : int
        Number of causes in the treatment.

    C : int
        Dimension of the latent factors.
    
    treatment_repr : TreatmentRepr
        Representation used for the treatment.

    Returns
    -------
    X : np.ndarray of shape (N, R * (M + H), D)
        Covariates, constructed from a linear component and a periodic component.
    
    M_ : np.ndarray of shape (N, R * (M + H), D)
        Masking vectors, which indicate whether a certain feature in covariates is measured.

    Y : np.ndarray of shape (N, M + H)
        Outcome measures, which span the pre-treatment and treatment time.
    
    A : np.ndarray of shape (N, K)
        Treatment indicator vectors, which indicate the cause configurations for each subject.
    """
    X = simulate_covariates(N, D // 4, D - D // 4, (M + H) * R)
    M_ = simulate_masking_vectors(N, D, (M + H) * R)
    A = simulate_treatment_vectors(N, K, treatment_repr)
    Y = simulate_outcomes(X, A, N, M, H, R, D, K, C)
    return X, M_, Y, A
