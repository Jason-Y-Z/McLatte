"""
Pharmacokinetic-Pharmacodynamic model for simulated data, adapted from
https://github.com/vanderschaarlab/SyncTwin-NeurIPS-2021
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

from dataclasses import dataclass

import numpy as np
import numpy.random
import scipy.integrate
import torch

from mclatte.synctwin._config import DEVICE
from mclatte.test_data import io_utils


def f(t, y, Kin, K, O, H, D50):
    P = y[0]
    R = y[1]
    D = y[2]

    dP = Kin[int(t)] - K * P
    dR = K * P - (D / (D + D50)) * K * R
    dD = O[int(t)] - H * D

    return [dP, dR, dD]


def solve(init, Kin, K, Os, H, D50, step=30):
    ode = scipy.integrate.ode(f).set_integrator("dopri5")

    Ot = np.zeros(step + 1)
    if Os >= 0:
        Ot[Os:] = 1.0

    try:
        len(Kin)
    except Exception:  # pylint: disable=broad-except
        Kin = np.ones(step + 1) * Kin

    ode.set_initial_value(init, 0).set_f_params(Kin, K, Ot, H, D50)
    t1 = step
    dt = 1

    res_list = []

    while ode.successful() and ode.t < t1:
        res = ode.integrate(ode.t + dt, ode.t + dt)
        res_list.append(res)

    res = np.stack(res_list, axis=-1)
    return res


def get_Kin(step=30, n_basis=12):
    # define Kin
    Kin_b_list = []
    Kin_b_list.append(np.ones(step + 1))
    x = np.arange(step + 1) / step
    Kin_b_list.append(x)

    for _ in range(n_basis - 2):
        bn = 2 * x * Kin_b_list[-1] - Kin_b_list[-2]
        Kin_b_list.append(bn)

    Kin_b = np.stack(Kin_b_list, axis=-1)

    Kin_list = [Kin_b[:, i] for i in range(n_basis)]
    return Kin_list, Kin_b


def get_clustered_Kin(Kin_b, n_cluster, n_sample_total):
    n_basis = Kin_b.shape[1]

    n_sample_cluster = n_sample_total // n_cluster
    if n_sample_total % n_cluster != 0:
        print("Warning: sample size not divisible by number of clusters")

    # generate cluster masks
    mask_list = []
    for i in range(n_cluster):
        mask = np.zeros(n_basis)
        mask[i:-1:4] = 1.0
        mask_list.append(mask)

    Kin_list = []
    for mask in mask_list:
        for _ in range(n_sample_cluster):
            Kin = np.matmul(Kin_b, numpy.random.randn(n_basis) * mask)
            Kin_list.append(Kin)

    Kin_b = np.stack(Kin_list, axis=-1)
    return Kin_list, Kin_b


def generate_control_data(
    Kin_list, K_list, P0_list, R0_list, train_step, H=0.1, D50=3, step=30
):

    # K_list = [0.18, 0.28, 0.38]
    # K_list = [0.18]

    # P0_list = [0., 0.5, 1.]
    # P0_list = [0.5]

    # R0_list = [0.5, 1., 1.5]
    # R0_list = [0.5]

    control_res_list = []

    for Kin in Kin_list:
        for K in K_list:
            for P0 in P0_list:
                for R0 in R0_list:
                    control_res = solve([P0, R0, 0.0], Kin, K, train_step, H, D50, step)
                    control_res_list.append(control_res)

    control_res_arr = np.stack(control_res_list, axis=-1)
    # Dim, T, N
    # Dim = 3: precursor, Cholesterol, Statins concentration
    # slice on dim=1 to get the outcome of interest
    return control_res_arr


def get_covariate(
    control_Kin_b,
    treat_Kin_b,
    control_res_arr,
    treat_res_arr,
    step=30,
    train_step=25,
    device=DEVICE,
    noise=None,
    double_up=False,
    hidden_confounder=0,
):
    n_units = control_res_arr.shape[-1] * 2 if double_up else control_res_arr.shape[-1]
    n_treated = treat_res_arr.shape[-1]

    covariates_control = np.concatenate(
        [control_Kin_b[:step, :][None, :, :], control_res_arr], axis=0
    )
    covariates_treated = np.concatenate(
        [treat_Kin_b[:step, :][None, :, :], treat_res_arr], axis=0
    )
    covariates = np.concatenate([covariates_control, covariates_treated], axis=2)

    covariates = torch.tensor(covariates, dtype=torch.float32)

    covariates = covariates.permute((1, 2, 0)).to(device)
    # remove the last covariate
    covariates = covariates[:, :, :3]

    # standardize
    m = covariates.mean(dim=(0, 1))
    sd = covariates.std(dim=(0, 1))
    covariates = (covariates - m) / sd

    if double_up:
        covariates_control = covariates[:, : (covariates.shape[1] // 2), :]
        covariates_twin = (
            covariates_control + torch.randn_like(covariates_control) * 0.1
        )
        covariates = torch.cat([covariates_twin, covariates], dim=1)

    if noise is not None:
        covariates = covariates + torch.randn_like(covariates) * noise

    n_units_total = n_units + n_treated

    pretreatment_time = train_step

    x_full = covariates[:pretreatment_time, :, :]
    if hidden_confounder == 1:
        x_full[:, :, 0] = 0
    if hidden_confounder == 2:
        x_full[:, :, 0] = 0
        x_full[:, :, 1] = 0
    y_full = covariates[pretreatment_time:, :, 2:3].detach().clone()
    y_full_all = covariates[pretreatment_time:, :, :]
    y_control = covariates[pretreatment_time:, :n_units, 2:3]

    t_full = torch.ones_like(x_full)
    mask_full = torch.ones_like(x_full)
    batch_ind_full = torch.arange(n_units_total).to(DEVICE)
    y_mask_full = (batch_ind_full < n_units) * 1.0
    return (
        (n_units, n_treated, n_units_total),
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,
        y_full_all,
        m,
        sd,
    )


def get_treatment_effect(
    treat_res_arr, treat_counterfactual_arr, train_step, m, sd, device=DEVICE
):
    m = m[2:3].item()
    sd = sd[2:3].item()

    treat_res_arr = (treat_res_arr - m) / sd
    treat_counterfactual_arr = (treat_counterfactual_arr - m) / sd

    return torch.tensor(
        treat_res_arr - treat_counterfactual_arr, device=device
    ).permute((1, 2, 0))[train_step:, :, 1:2]


@dataclass
class PkpdDataGenConfig:
    """
    Configurations for PKPD data generation.
    """

    n: int = 200
    p_0: str = "0.1"
    seed: int = 509
    m: int = 5
    h: int = 5
    r: int = 5
    d: int = 3
    k: int = 1
    c: int = 3


def generate_data(constants: PkpdDataGenConfig, return_raw=True):
    """
    Generate a training and testing dataset of PKPD.
    """
    base_path_data = f"data/pkpd/{constants.p_0}_{constants.n}-seed-{constants.seed}"
    data_path = base_path_data + "/{}-{}.{}"

    # loading config and data
    io_utils.load_config(data_path, "train")
    x_full, t_full, mask_full, _, y_full, _, _, _, _, _ = io_utils.load_tensor(
        data_path, "train", device="cpu"
    )
    (
        x_full_val,
        t_full_val,
        mask_full_val,
        _,
        y_full_val,
        _,
        _,
        _,
        _,
        _,
    ) = io_utils.load_tensor(data_path, "val", device="cpu")

    x = np.concatenate((x_full.cpu().numpy(), x_full_val.cpu().numpy()), axis=1)
    t = np.concatenate((t_full.cpu().numpy(), t_full_val.cpu().numpy()), axis=1)
    mask = np.concatenate(
        (mask_full.cpu().numpy(), mask_full_val.cpu().numpy()), axis=1
    )
    y = np.concatenate(
        (y_full.cpu().numpy(), y_full_val.cpu().numpy()), axis=1
    ).squeeze()

    X = x.transpose((1, 0, 2))
    N = X.shape[0]
    rand_index = np.random.permutation(N)
    X = X[rand_index]
    M_ = mask.transpose((1, 0, 2))[rand_index]
    Y_pre = y.T[rand_index]
    Y_post = y.T[rand_index]
    A = np.concatenate(
        (
            np.zeros((N // 4, 1)),
            np.ones((N // 4, 1)),
            np.zeros((N // 4, 1)),
            np.ones((N // 4, 1)),
        ),
        axis=0,
    )[rand_index]
    T = t.transpose((1, 0, 2))[rand_index]

    N_train = round(N * 0.8)
    N_test = round(N * 0.2)
    X_train, X_test = X[:N_train], X[N_train:]
    M_train, M_test = M_[:N_train], M_[N_train:]
    Y_pre_train, Y_pre_test = Y_pre[:N_train], Y_pre[N_train:]
    Y_post_train, Y_post_test = Y_post[:N_train], Y_post[N_train:]
    A_train, A_test = A[:N_train], A[N_train:]
    T_train, T_test = T[:N_train], T[N_train:]

    all_data = (
        (
            N_train,
            constants.m,
            constants.h,
            constants.r,
            constants.d,
            constants.k,
            constants.c,
            X_train,
            M_train,
            Y_pre_train,
            Y_post_train,
            A_train,
            T_train,
        ),
        (
            N_test,
            constants.m,
            constants.h,
            constants.r,
            constants.d,
            constants.k,
            constants.c,
            X_test,
            M_test,
            Y_pre_test,
            Y_post_test,
            A_test,
            T_test,
        ),
    )

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
    return N, train_data, test_data
