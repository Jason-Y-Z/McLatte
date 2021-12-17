"""
Utilities for plotting model testing results.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import os

import joblib
import numpy as np
import plotly.graph_objects as go

from mclatte.mclatte.model import (
    infer_mcespresso,
    McLatte,
    SemiSkimmedMcLatte,
    SkimmedMcLatte,
)
from mclatte.rnn.model import (
    infer_rnn,
    BaselineRnn,
)
from mclatte.synctwin.model import (
    infer_synctwin,
    SyncTwinPl,
)


PLOT_NAME = {
    "skimmed_mclatte": "S",
    "semi_skimmed_mclatte": "SS",
    "mclatte": "V",
    "rnn": "RNN",
    "synctwin": "SyncTwin",
}


def print_losses(config_idx):
    all_losses = joblib.load(f"results/test/config_{config_idx}_idt.joblib")
    for losses in all_losses:
        print(f"{np.mean(losses):.3f} ({np.std(losses):.3f})")


def get_y_tildes(
    data_name,
    config_id,
    test_data,
):
    def get_ckpt_path(model_name):
        return os.path.join(
            os.getcwd(),
            f"results/{data_name}/trained_models/{model_name}_{config_id + 1}.ckpt",
        )

    y_tildes = []

    mclatte_args = test_data.x, test_data.a, test_data.t, test_data.m
    trained_model = SkimmedMcLatte.load_from_checkpoint(
        get_ckpt_path("skimmed_mclatte")
    )
    y_tildes.append(
        (infer_mcespresso(trained_model, *mclatte_args)[1], "skimmed_mclatte")
    )
    trained_model = SemiSkimmedMcLatte.load_from_checkpoint(
        get_ckpt_path("semi_skimmed_mclatte")
    )
    y_tildes.append(
        (infer_mcespresso(trained_model, *mclatte_args)[2], "semi_skimmed_mclatte")
    )
    trained_model = McLatte.load_from_checkpoint(get_ckpt_path("mclatte"))
    y_tildes.append((infer_mcespresso(trained_model, *mclatte_args)[2], "mclatte"))

    trained_model = BaselineRnn.load_from_checkpoint(get_ckpt_path("rnn"))
    y_tildes.append(
        (
            infer_rnn(
                trained_model, test_data.y_pre, test_data.y_post, return_Y_pred=True
            ),
            "rnn",
        )
    )

    trained_model = SyncTwinPl.load_from_checkpoint(get_ckpt_path("synctwin"))
    y_tildes.append(
        (infer_synctwin(trained_model, test_data.n, test_data.y_post), "synctwin")
    )

    return y_tildes


def plot_subject(y_tildes, plot_sub_id, y_pre_plot, post_t, file_suffix, fig):
    for y_tilde, model_name in y_tildes:
        y_pred_plot = [y_pre_plot[-1]] + list(y_tilde.detach().numpy()[plot_sub_id])
        line_pred_model = go.Scatter(
            x=post_t,
            y=y_pred_plot,
            name=f"{PLOT_NAME[model_name]}{file_suffix}",
            line={"dash": "dash"},
        )
        fig.add_trace(line_pred_model)


def plot_config_results(
    data_name,
    test_configs,
    generate_data,
    config_id,
    file_suffix="",
    plot_01_treatment=True,
):
    config = test_configs[config_id]
    _, _, test_data = generate_data(*config, return_raw=False)
    y_tildes = get_y_tildes(data_name, config, test_data)

    for plot_sub_id in range(test_data.n):
        y_pre_plot = test_data.y_pre[plot_sub_id]
        pre_t = list(np.arange(y_pre_plot.shape[0]) - y_pre_plot.shape[0])

        y_post_plot = [y_pre_plot[-1]] + list(test_data.y_post[plot_sub_id])
        post_t = np.arange(len(y_post_plot))

        trt_str = ", ".join(
            map(
                lambda x: str(round(x, 2)) if abs(x - round(x)) > 5e-2 else str(int(x)),
                test_data.a[plot_sub_id],
            )
        )

        fig = go.Figure()
        line_pre_trt = go.Scatter(
            x=pre_t + list(post_t),
            y=list(y_pre_plot) + y_post_plot,
            name="ground truth",
        )
        fig.add_trace(line_pre_trt)

        plot_subject(y_tildes, plot_sub_id, y_pre_plot, post_t, file_suffix, fig)
        if plot_01_treatment:
            test_data.a[plot_sub_id] = (
                np.ones_like(test_data.a[plot_sub_id])
                if not (test_data.a[plot_sub_id] == 0).all()
                else np.zeros_like(test_data.a[plot_sub_id])
            )
            plot_subject(
                y_tildes, plot_sub_id, y_pre_plot, post_t, file_suffix + " 01", fig
            )

        fig.update_layout(
            title=f"Outcome for Treatment Vector ({trt_str})",
            yaxis_title="Outcome",
            xaxis_title="Time",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        fig.write_image(
            f"plots/{data_name}/outcome_pred_{config_id}_{plot_sub_id}{file_suffix}.png"
        )
