"""
Application script for running SyncTwin model training.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import argparse
import os

import joblib
import numpy as np
import ray
from ray import tune
import wandb

from mclatte.synctwin.model import train_synctwin


def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser("SyncTwin training")
    parser.add_argument("--data", type=str, default="diabetes")
    args = parser.parse_args()

    # Initialising environment
    wandb.init(project="mclatte-test", entity="jasonyz")
    np.random.seed(509)
    ray.init(address=None)

    # Load model training dataset
    N, M, _, R, D, _, _, X, M_, _, Y_post, A, T = joblib.load(
        os.path.join(os.getcwd(), f"data/{args.data}/hp_search.joblib")
    )
    Y_mask = np.all(A == 0, axis=1)
    Y_control = Y_post[Y_mask]

    # Initialise hyper-parameter search space
    hp_config = {
        "hidden_dim": tune.choice([8, 32, 128]),
        "reg_B": tune.uniform(0, 1),
        "lam_express": tune.uniform(0, 1),
        "lam_recon": tune.uniform(0, 1),
        "lam_prognostic": tune.uniform(0, 1),
        "tau": tune.uniform(0, 1),
        "batch_size": tune.choice([32]),
        "epochs": tune.choice([100]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "gamma": tune.uniform(0.5, 0.99),
    }
    sync_config = tune.SyncConfig()

    # Run hyper-parameter search
    sync_config = tune.SyncConfig()
    st_trainable = tune.with_parameters(
        train_synctwin,
        X=X,
        M_=M_,
        T=T,
        Y_batch=Y_post,
        Y_control=Y_control,
        Y_mask=Y_mask,
        N=N,
        D=D,
        n_treated=N - Y_control.shape[0],
        pre_trt_x_len=R * M,
    )
    analysis = tune.run(
        st_trainable,
        name="tune_pl_sync_twin",
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

    # Save results
    analysis.results_df.to_csv(os.path.join(os.getcwd(), "results/synctwin_hp.csv"))


if __name__ == "__main__":
    main()
