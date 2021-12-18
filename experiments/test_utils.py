"""
Utilities for testing models with best hyperparameters.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind

from mclatte.mclatte.model import (
    infer_mcespresso,
    train_mclatte,
    train_semi_skimmed_mclatte,
    train_skimmed_mclatte,
)
from mclatte.rnn.model import (
    infer_rnn,
    train_baseline_rnn,
)
from mclatte.synctwin.model import (
    train_synctwin,
)


def test_skimmed_mclatte(
    skimmed_mclatte_config,
    constants,
    train_data,
    test_data,
    run_idx=0,
):
    trained_skimmed_mclatte = train_skimmed_mclatte(
        skimmed_mclatte_config,
        constants,
        train_data,
        test_run=run_idx,
    )
    _, y_tilde = infer_mcespresso(
        trained_skimmed_mclatte, test_data["x"], test_data["a"], test_data["t"], test_data["m"]
    )

    return torch.nn.functional.l1_loss(
        y_tilde, torch.from_numpy(test_data["y_post"]).float()
    ).item()


def test_semi_skimmed_mclatte(
    semi_skimmed_mclatte_config,
    constants,
    train_data,
    test_data,
    run_idx=0,
):
    trained_semi_skimmed_mclatte = train_semi_skimmed_mclatte(
        semi_skimmed_mclatte_config,
        constants,
        train_data,
        test_run=run_idx,
    )
    _, _, y_tilde = infer_mcespresso(
        trained_semi_skimmed_mclatte, test_data["x"], test_data["a"], test_data["t"], test_data["m"]
    )

    return torch.nn.functional.l1_loss(
        y_tilde, torch.from_numpy(test_data["y_post"]).float()
    ).item()


def test_mclatte(
    mclatte_config,
    constants,
    train_data,
    test_data,
    run_idx=0,
):
    trained_mclatte = train_mclatte(
        mclatte_config,
        constants,
        train_data,
        test_run=run_idx,
    )
    _, _, y_tilde = infer_mcespresso(
        trained_mclatte, test_data["x"], test_data["a"], test_data["t"], test_data["m"]
    )

    return torch.nn.functional.l1_loss(
        y_tilde, torch.from_numpy(test_data["y_post"]).float()
    ).item()


def test_rnn(
    rnn_config,
    train_data,
    test_data,
    run_idx=0,
):
    trained_rnn = train_baseline_rnn(
        rnn_config,
        Y=np.concatenate((train_data["y_pre"], train_data["y_post"]), axis=1),
        input_dim=1,
        test_run=run_idx,
    )
    return infer_rnn(trained_rnn, test_data["y_pre"], test_data["y_post"])


def test_synctwin(
    synctwin_config,
    constants,
    train_data,
    test_data,
    run_idx=0,
):
    train_data["y_mask"] = np.all(train_data["a"] == 0, axis=1)
    test_data["y_mask"] = np.all(test_data["a"] == 0, axis=1)
    train_data["y_control"] = train_data["y_post"][train_data["y_mask"]]

    trained_synctwin = train_synctwin(
        synctwin_config,
        constants,
        train_data,
        test_run=run_idx,
    ).cpu()

    trained_synctwin.eval()
    _, l1_loss = trained_synctwin(
        torch.from_numpy(test_data["x"]).float().cpu(),
        torch.from_numpy(test_data["t"]).float().cpu(),
        torch.from_numpy(test_data["m"]).float().cpu(),
        torch.arange(0, test_data["n"]).cpu(),
        torch.from_numpy(test_data["y_post"]).float().cpu(),
        torch.from_numpy(test_data["y_mask"]).float().cpu(),
    )
    return l1_loss.item()


def test_losses(losses, loss_names):
    t_test_results = pd.DataFrame(columns=loss_names, index=loss_names)

    for i, loss_i in enumerate(losses):
        for j, loss_j in enumerate(losses):
            t = ttest_ind(loss_i, loss_j, alternative="less")
            t_test_results[loss_names[i]][loss_names[j]] = t.pvalue
    return t_test_results
