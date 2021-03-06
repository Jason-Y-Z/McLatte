"""
Data generation utilities for the simulation study.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import enum
from dataclasses import dataclass
from typing import Callable, Tuple, Union

import joblib
import numpy as np


def _logit(x):
    return 1.0 / (1 + np.exp(-x))


def _sample_for_N_T(N: int, T: int, sample: Callable) -> np.ndarray:
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
    return_conf: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
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

    return_conf : bool
        Whether to return the confounding parameters.

    Returns
    -------
    X : np.ndarray of shape (N, T, L + P)
        A multivariate normal time series,
        constructed from a linear component and a periodic component.

    mu_conf : float; only if return_conf is True
        Temporal confounding parameter.
    """
    t = np.arange(T)

    # Initialise parameters for the linear component
    alpha_linear = np.random.uniform(-100 / T, 100 / T, (N, L))
    beta_linear = np.random.uniform(-10, 10, (N, L))
    mu_linear = alpha_linear[:, None, :] * t[None, :, None] + beta_linear[:, None, :]
    sigma_linear = np.einsum(
        "ijk,ilk->ijl", (comp_max := np.random.uniform(-1, 1, (N, L, L))), comp_max
    )

    # Initialise parameters for the periodic component
    alpha_periodic = np.random.uniform(-10, 10, (N, P))
    beta_periodic = np.random.uniform(-10, 10, (N, P))
    omega_periodic = np.random.uniform(-np.pi, np.pi, (N, P))
    mu_periodic = (
        alpha_periodic[:, None, :]
        * np.sin(omega_periodic[:, None, :] * t[None, :, None])
        + beta_periodic[:, None, :]
    )
    sigma_periodic = np.einsum(
        "ijk,ilk->ijl", (comp_max := np.random.uniform(-1, 1, (N, P, P))), comp_max
    )

    # Calculate confounding parameters
    mu_conf = _logit(
        np.linalg.norm(alpha_linear)
        + np.linalg.norm(alpha_periodic)
        + np.linalg.norm(beta_linear)
        + np.linalg.norm(beta_periodic)
    )

    # Draw samples from the distributions defined by the initialised parameters
    def sample_x(i: int, t: int) -> np.ndarray:
        x_linear = np.random.multivariate_normal(mu_linear[i, t], sigma_linear[i])
        x_periodic = np.random.multivariate_normal(mu_periodic[i, t], sigma_periodic[i])
        x_it = np.concatenate([x_linear, x_periodic])
        return x_it

    samples = _sample_for_N_T(N, T, sample_x)
    return (samples, mu_conf) if return_conf else samples


def simulate_covariates(
    N: int,
    L: int,
    P: int,
    T: int,
    return_conf: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
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

    return_conf : bool
        Whether to return the confounding parameters.

    Returns
    -------
    X : np.ndarray of shape (N, T, L + P)
        Covariates, constructed from a linear component and a periodic component.

    mu_conf : float; only if return_conf is True
        Temporal confounding parameter.
    """
    return _simulate_multivariate_time_series(N, L, P, T, return_conf)


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
    return _sample_for_N_T(
        N, T, lambda i, _: np.random.binomial(np.ones(D, dtype=np.int32), p[i])
    )


class TreatmentRepr(enum.Enum):
    """
    Treatment representation types:
        binary - 0/1
        bounded - Uniform(a, b)
        real-valued - Normal(mu, sigma^2)
    """

    BINARY = 0
    BOUNDED = 1
    REAL_VALUED = 2


def simulate_treatment_vectors(
    N: int,
    K: int,
    mode: TreatmentRepr,
    p_0: float = 0,
    mu_conf: float = 0,
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

    p_0 : float
        Temporal confounding coefficient.

    mu_conf : float
        Temporal confounding parameter.

    Returns
    -------
    A : np.ndarray of shape (N, K)
        Treatment indicator vectors, which indicate the cause configurations for each subject.
    """
    if mode is TreatmentRepr.BINARY:
        all_data = np.concatenate(
            (
                np.stack(
                    [
                        np.random.binomial(
                            np.ones(K, dtype=np.int32),
                            _logit(np.ones(K) / 2 + p_0 * mu_conf),
                        )
                        for _ in range(round(N * 0.8))
                    ]
                ),
                np.zeros(
                    (round(N * 0.2), K)
                ),  # make sure there are enough control subjects
            ),
            axis=0,
        )
        np.random.shuffle(all_data)
        return all_data
    if mode is TreatmentRepr.BOUNDED:
        all_data = np.concatenate(
            (
                np.random.uniform(0, 1 + p_0 * mu_conf, (round(N * 0.8), K)),
                np.zeros(
                    (round(N * 0.2), K)
                ),  # make sure there are enough control subjects
            ),
            axis=0,
        )
        np.random.shuffle(all_data)
        return all_data
    if mode is TreatmentRepr.REAL_VALUED:
        all_data = np.concatenate(
            (
                np.random.normal(p_0 * mu_conf, 1, (round(N * 0.8), K)),
                np.zeros(
                    (round(N * 0.2), K)
                ),  # make sure there are enough control subjects
            ),
            axis=0,
        )
        np.random.shuffle(all_data)
        return all_data
    raise ValueError(f"Unrecognized treatment representation: mode = {mode}")


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
    return_c: bool = False,
):
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

    return_c : bool
        Whether to return the latent factors.

    Returns
    -------
    Y_pre : np.ndarray of shape (N, M)
        Pre-treatment outcome measures.

    Y_post : np.ndarray of shape (N, H)
        Post-treatment outcome measures.

    C_post : np.ndarray of shape (N, C); only if return_c is True
        Post-treatment latent factors.
    """
    X_pre, X_post = X[:, : M * R, :], X[:, M * R :, :]

    g = np.random.rand(1, M * R)
    G = np.random.rand(K, H * R) / H
    H_ = np.random.rand(D, C) * 1e-3

    C_pre = np.stack([np.squeeze(g @ X_pre[i] @ H_) for i in range(N)])
    C_post = np.stack([np.squeeze(A[i] @ G @ X_post[i] @ H_) for i in range(N)])
    Q = _simulate_multivariate_time_series(1, C // 3, C - C // 3, M + H).reshape(
        (M + H, C)
    )

    Y_pre = _sample_for_N_T(
        N, M, lambda i, t: np.random.normal(np.dot(Q[t], C_pre[i]), 1)
    ).reshape((N, M))
    Y_post = _sample_for_N_T(
        N, H, lambda i, t: np.random.normal(np.dot(Q[t], C_post[i]), 1)
    ).reshape((N, H))
    if return_c:
        return Y_pre, Y_post, C_post
    return Y_pre, Y_post


def generate_simulation_data(
    N: int,
    M: int,
    H: int,
    R: int,
    D: int,
    K: int,
    C: int,
    treatment_repr: TreatmentRepr,
    p_0: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
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

    p_0 : float
        Temporal confounding coefficient.

    Returns
    -------
    X : np.ndarray of shape (N, R * M, D)
        Covariates, constructed from a linear component and a periodic component.

    M_ : np.ndarray of shape (N, R * M, D)
        Masking vectors, which indicate whether a certain feature in covariates is measured.

    Y_pre : np.ndarray of shape (N, M)
        Pre-treatment outcome measures.

    Y_post : np.ndarray of shape (N, H)
        Post-treatment outcome measures.

    A : np.ndarray of shape (N, K)
        Treatment indicator vectors, which indicate the cause configurations for each subject.

    T : np.ndarray of shape (N, R * M, D)
        Measurement time for each quantity.
    """
    X, mu_conf = simulate_covariates(
        N, D // 4, D - D // 4, (M + H) * R, return_conf=True
    )
    M_ = simulate_masking_vectors(N, D, (M + H) * R)
    A = simulate_treatment_vectors(N, K, treatment_repr, p_0, mu_conf)
    Y_pre, Y_post = simulate_outcomes(  # pylint: disable=unbalanced-tuple-unpacking
        X, A, N, M, H, R, D, K, C
    )
    T = np.repeat(
        np.repeat(
            np.arange(-R * M, R * H).reshape((1, -1, 1)),
            repeats=D,
            axis=2,
        ),
        repeats=N,
        axis=0,
    )
    return X[:, : M * R, :], M_[:, : M * R, :], Y_pre, Y_post, A, T[:, : M * R, :]


@dataclass
class SimDataGenConfig:
    """
    Configurations for simulated data generation.
    """

    n: int = 200
    p_0: float = 0.1
    mode: TreatmentRepr = TreatmentRepr.BINARY
    m: int = 5
    h: int = 5
    r: int = 5
    d: int = 3
    k: int = 4
    c: int = 8


def generate_data(
    constants: SimDataGenConfig, run_idx: int = 0, return_raw: bool = True
):
    """
    Generate a training and testing dataset of idealised disease treatment.
    """
    data_path = (
        f"data/test/idt_{constants.n}_{constants.p_0}_{constants.mode}_{run_idx}.joblib"
    )
    try:
        all_data = joblib.load(data_path)
        (
            _,
            N_train,
            N_test,
            X_train,
            X_test,
            M_train,
            M_test,
            Y_pre_train,
            Y_pre_test,
            Y_post_train,
            Y_post_test,
            A_train,
            A_test,
            T_train,
            T_test,
        ) = all_data
    except Exception as e:  # pylint: disable=broad-except
        print(e)
        N_train = round(constants.n * 0.8)
        N_test = round(constants.n * 0.2)
        X, M_, Y_pre, Y_post, A, T = generate_simulation_data(
            constants.n,
            constants.m,
            constants.h,
            constants.r,
            constants.d,
            constants.k,
            constants.c,
            constants.mode,
            constants.p_0,
        )
        X_train, X_test = X[:N_train], X[N_train:]
        M_train, M_test = M_[:N_train], M_[N_train:]
        Y_pre_train, Y_pre_test = Y_pre[:N_train], Y_pre[N_train:]
        Y_post_train, Y_post_test = Y_post[:N_train], Y_post[N_train:]
        A_train, A_test = A[:N_train], A[N_train:]
        T_train, T_test = T[:N_train], T[N_train:]
        all_data = (
            constants.n,
            N_train,
            N_test,
            X_train,
            X_test,
            M_train,
            M_test,
            Y_pre_train,
            Y_pre_test,
            Y_post_train,
            Y_post_test,
            A_train,
            A_test,
            T_train,
            T_test,
        )
        joblib.dump(all_data, data_path)

    if return_raw:
        return all_data

    train_data = dict(
        n=N_train,
        x=X_train,
        m=M_train,
        y_pre=Y_pre_train,
        y_post=Y_post_train,
        a=A_train,
        t=T_train,
    )
    test_data = dict(
        n=N_test,
        x=X_test,
        m=M_test,
        y_pre=Y_pre_test,
        y_post=Y_post_test,
        a=A_test,
        t=T_test,
    )
    return constants.n, train_data, test_data
