""" 
Application script for running McLatte Whole model training.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import argparse
import joblib
import numpy as np
import os
import ray
import wandb
from mclatte.model import train_mclatte
from ray import tune


def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser('McLatte training')
    parser.add_argument('--data', type=str, default='pkpd')
    args = parser.parse_args()

    # Initialising environment
    wandb.init(project='mclatte-test', entity='jasonyz')
    np.random.seed(509)
    ray.init(address=None)

    # Load model training dataset
    _, M, H, R, D, K, _, X, M_, Y_pre, Y_post, A, T = joblib.load(
        os.path.join(os.getcwd(), f'data/{args.data}/data_0.25_200.joblib')
    )

    # Initialise hyper-parameter search space
    hp_config = {
        'encoder_class': tune.choice(['lstm']),
        'decoder_class': tune.choice(['lstm']),
        'hidden_dim': tune.choice([4, 16, 64]),
        'batch_size': tune.choice([64]),
        'epochs': tune.choice([100]),
        'lr': tune.loguniform(1e-4, 1e-0),
        'gamma': tune.uniform(0.5, 0.99),
        'lambda_r': tune.loguniform(1e-2, 1e2),
        'lambda_d': tune.loguniform(1e-2, 1e2),
        'lambda_p': tune.loguniform(1e-2, 1e2),
    }

    # Run hyper-parameter search
    sync_config = tune.SyncConfig()
    mclatte_trainable = tune.with_parameters(
        train_mclatte,
        X=X,
        M_=M_,
        Y_pre=Y_pre,
        Y_post=Y_post,
        A=A, 
        T=T,
        R=R,
        M=M,
        H=H,
        input_dim=D, 
        treatment_dim=K,
    )
    analysis = tune.run(
        mclatte_trainable,
        name='tune_pl_mclatte',
        local_dir=os.path.join(os.getcwd(), 'data'),
        sync_config=sync_config,
        resources_per_trial={
            'cpu': 4,
        },
        metric='valid_loss',
        mode='min',
        checkpoint_score_attr='valid_loss',
        keep_checkpoints_num=5,
        config=hp_config,
        num_samples=20,
        verbose=1,
        resume='AUTO',
    )

    # Save results
    analysis.results_df.to_csv(os.path.join(os.getcwd(), 'results/mclatte_hp_pkpd.csv'))


if __name__ == '__main__':
    main()
