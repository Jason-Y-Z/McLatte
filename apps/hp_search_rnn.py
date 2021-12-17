""" 
Application script for running baseline RNN model training.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import argparse
import joblib
import numpy as np
import os
import ray
import wandb
from mclatte.rnn.model import train_baseline_rnn
from ray import tune


def main():
    parser = argparse.ArgumentParser("RNN training")
    parser.add_argument("--data", type=str, default="diabetes")
    args = parser.parse_args()

    wandb.init(project="mclatte-test", entity="jasonyz")
    np.random.seed(509)
    ray.init(address=None)

    _, _, _, _, _, _, _, _, _, Y_pre, Y_post, _, _ = joblib.load(
        os.path.join(os.getcwd(), f"data/{args.data}/hp_search.joblib")
    )

    hp_config = {
        "rnn_class": tune.choice(["rnn", "lstm", "gru"]),
        "hidden_dim": tune.choice([4, 16, 64]),
        "seq_len": tune.choice([2, 4]),
        "batch_size": tune.choice([64]),
        "epochs": tune.choice([100]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "gamma": tune.uniform(0.5, 0.99),
    }
    sync_config = tune.SyncConfig()
    rnn_trainable = tune.with_parameters(
        train_baseline_rnn,
        Y=np.concatenate((Y_pre, Y_post), axis=1),
        input_dim=1,
    )

    analysis = tune.run(
        rnn_trainable,
        name="tune_pl_baseline_rnn",
        local_dir=os.path.join(os.getcwd(), "data"),
        sync_config=sync_config,
        resources_per_trial={
            "cpu": 4,
            "gpu": 0,
        },
        metric="valid_loss",
        mode="min",
        checkpoint_score_attr="valid_loss",
        keep_checkpoints_num=5,
        config=hp_config,
        num_samples=20,
        verbose=1,
        resume="AUTO",
    )
    analysis.results_df.to_csv(
        os.path.join(os.getcwd(), "results/baseline_rnn_hp.csv")
    )


if __name__ == "__main__":
    main()
