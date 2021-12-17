import numpy as np
import pandas as pd
import torch
from mclatte.mclatte.model import (
    train_mclatte,
    train_semi_skimmed_mclatte,
    train_skimmed_mclatte,
)
from mclatte.rnn.model import (
    train_baseline_rnn,
)
from mclatte.synctwin.model import (
    train_synctwin,
)
from scipy.stats import ttest_ind


def infer_mcespresso(trained_mcespresso, x, a, t, m):
    trained_mcespresso.eval()
    return trained_mcespresso(
        torch.from_numpy(x).float(),
        torch.from_numpy(a).float(),
        torch.from_numpy(t).float(),
        torch.from_numpy(m).float(),
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
        trained_skimmed_mclatte, test_data.x, test_data.a, test_data.t, test_data.m
    )

    return torch.nn.functional.l1_loss(
        y_tilde, torch.from_numpy(test_data.y_post).float()
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
        trained_semi_skimmed_mclatte, test_data.x, test_data.a, test_data.t, test_data.m
    )

    return torch.nn.functional.l1_loss(
        y_tilde, torch.from_numpy(test_data.y_post).float()
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
        trained_mclatte, test_data.x, test_data.a, test_data.t, test_data.m
    )

    return torch.nn.functional.l1_loss(
        y_tilde, torch.from_numpy(test_data.y_post).float()
    ).item()


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


def test_rnn(
    rnn_config,
    train_data,
    test_data,
    run_idx=0,
):
    trained_rnn = train_baseline_rnn(
        rnn_config,
        Y=np.concatenate((train_data.y_pre, train_data.y_post), axis=1),
        input_dim=1,
        test_run=run_idx,
    )
    return infer_rnn(trained_rnn, test_data.y_pre, test_data.y_post)


def infer_synctwin(trained_synctwin, N_test, Y_post_test):
    trained_synctwin.eval()
    return trained_synctwin._sync_twin.get_prognostics(
        torch.arange(0, N_test).cpu(), torch.from_numpy(Y_post_test).float().cpu()
    )


def test_synctwin(
    synctwin_config,
    constants,
    train_data,
    test_data,
    run_idx=0,
):
    train_data["y_mask"] = np.all(train_data.a == 0, axis=1)
    test_data["y_mask"] = np.all(test_data.a == 0, axis=1)
    train_data["y_control"] = train_data.y_post[train_data.y_mask]

    trained_synctwin = train_synctwin(
        synctwin_config,
        constants,
        train_data,
        test_run=run_idx,
    ).cpu()

    trained_synctwin.eval()
    _, l1_loss = trained_synctwin(
        torch.from_numpy(test_data.x).float().cpu(),
        torch.from_numpy(test_data.t).float().cpu(),
        torch.from_numpy(test_data.m).float().cpu(),
        torch.arange(0, test_data.n).cpu(),
        torch.from_numpy(test_data.y_post).float().cpu(),
        torch.from_numpy(test_data.y_mask).float().cpu(),
    )
    return l1_loss.item()


def test_losses(losses, loss_names):
    t_test_results = pd.DataFrame(columns=loss_names, index=loss_names)

    for i in range(len(losses)):
        for j in range(len(losses)):
            t = ttest_ind(losses[i], losses[j], alternative="less")
            t_test_results[loss_names[i]][loss_names[j]] = t.pvalue
    return t_test_results
