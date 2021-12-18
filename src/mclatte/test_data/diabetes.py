"""
Data generation utilities for the Diabetes study.
"""
# Author: Jason Zhang (yurenzhang2017@gmail.com)
# License: BSD 3 clause

import datetime
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale


N_SUBJECTS = 70
M = 5
H = 5
R = 5
DATA_CODES = {
    33: "reg_insulin",  # treatment
    34: "nph_insulin",  # treatment
    35: "ult_insulin",  # treatment
    48: "unspecified_bg",  # outcome
    57: "unspecified_bg",  # outcome
    58: "pre_breakfast_bg",  # outcome
    59: "post_breakfast_bg",  # outcome
    60: "pre_lunch_bg",  # outcome
    61: "post_lunch_bg",  # outcome
    62: "pre_supper_bg",  # outcome
    63: "post_supper_bg",  # outcome
    64: "pre_snack_bg",  # outcome
    65: "hypo_symptoms",  # covariate
    66: "typical_meal",  # covariate
    67: "more_meal",  # covariate
    68: "less_meal",  # covariate
    69: "typical_exercise",  # covariate
    70: "more_exercise",  # covariate
    71: "less_exercise",  # covariate
    72: "unspecified_event",  # covariate
}
TREATMENT_COLS = ["reg_insulin", "nph_insulin", "ult_insulin"]
OUTCOME_COLS = [
    "unspecified_bg",
    "pre_breakfast_bg",
    "post_breakfast_bg",
    "pre_lunch_bg",
    "post_lunch_bg",
    "pre_supper_bg",
    "post_supper_bg",
    "pre_snack_bg",
]


def float_or_na(v):
    try:
        return float(v)
    except Exception as e:  # pylint: disable=broad-except
        if v == "0Hi":
            return 1
        if v == "0Lo":
            return -1
        print(e)
        return np.nan


def combine(values):
    valid_values = values[pd.notna(values)]
    if valid_values.shape[0] == 0:
        return np.nan
    return np.median(valid_values)


def try_to_date(v):
    try:
        return datetime.datetime.strptime(v, "%m-%d-%Y")
    except Exception as e:  # pylint: disable=broad-except
        print(f"{e}: {v}")
    try:
        v = v[:4] + "0" + v[5:]  # handle date mis-input (e.g. 6-31)
        return datetime.datetime.strptime(v, "%m-%d-%Y")
    except Exception as e:  # pylint: disable=broad-except
        print(f"{e}: {v}")
        return np.nan


def try_to_time(v):
    try:
        return datetime.datetime.strptime(v, "%H:%M").time()
    except Exception as e:  # pylint: disable=broad-except
        print(f"{e}: {v}")
        return np.nan


def try_to_combine(date, time):
    try:
        return datetime.datetime.combine(date, time)
    except Exception as e:  # pylint: disable=broad-except
        print(f"{e}: {date} {time}")
    if isinstance(date, datetime.datetime):
        return date
    return np.nan


def load_subject_i(subject_idx):
    raw_df = pd.read_csv(
        os.path.join(os.getcwd(), f"data/diabetes/data-{subject_idx:02d}"),
        sep="\t",
        names=["date", "time", "code", "value"],
    )
    raw_df["date"] = raw_df["date"].apply(try_to_date)
    raw_df["time"] = raw_df["time"].apply(try_to_time)
    raw_df["datetime"] = raw_df.apply(
        lambda row: try_to_combine(row["date"], row["time"]), axis=1
    )
    raw_df.drop(columns=["date", "time"], inplace=True)
    raw_df.sort_values(by=["datetime"], inplace=True)

    all_datetimes = raw_df.datetime.values
    converted_df = pd.DataFrame(
        index=range(len(set(all_datetimes))), columns=list(DATA_CODES.values())
    )

    begin_idx = 0
    converted_idx = 0
    while begin_idx < raw_df.shape[0]:
        while begin_idx < raw_df.shape[0] and np.isnan(all_datetimes[begin_idx]):
            begin_idx += 1

        end_idx = begin_idx
        while (
            end_idx < raw_df.shape[0]
            and all_datetimes[end_idx] == all_datetimes[begin_idx]
        ):
            if raw_df.iloc[end_idx]["code"] in DATA_CODES:
                col_name = DATA_CODES[raw_df.iloc[end_idx]["code"]]
                converted_df.iloc[converted_idx][col_name] = float_or_na(
                    raw_df.iloc[end_idx]["value"]
                )
            end_idx += 1
        begin_idx = end_idx
        converted_idx += 1

    outcomes = converted_df.apply(
        lambda row: combine(row[OUTCOME_COLS]), axis=1
    )  # pylint: disable=unnecessary-lambda
    treatment = converted_df[TREATMENT_COLS].apply(lambda col: combine(col), axis=0)
    converted_df = converted_df[TREATMENT_COLS + OUTCOME_COLS]

    mask_df = ~converted_df.isna()
    converted_df[pd.isna(converted_df)] = 0
    treatment[pd.isna(treatment)] = 0
    return (
        converted_df.to_numpy(),
        mask_df.to_numpy(),
        outcomes[pd.notna(outcomes)].to_numpy(),
        treatment.to_numpy(),
    )


def load_data():
    # Initialisation
    X = []
    M_ = []
    Y_pre = []
    Y_post = []
    A = []

    # Reading
    for subject_idx in range(1, N_SUBJECTS + 1):
        X_i, M_i, Y_i, A_i = load_subject_i(subject_idx)

        if M * R > M_i.shape[0]:
            M_.append(
                np.concatenate((np.zeros((M * R - M_i.shape[0], M_i.shape[1])), M_i))
            )
        else:
            M_.append(M_i[-M * R :])

        if H + M > Y_i.shape[0]:
            Y_pre.append(np.concatenate((np.zeros(H + M - Y_i.shape[0]), Y_i[:-H])))
        else:
            Y_pre.append(Y_i[-(H + M) : -H])

        Y_post.append(Y_i[-H:])
        A.append(A_i)
        X_i = X_i[:-H]
        if M * R > X_i.shape[0]:
            X.append(
                np.concatenate((np.zeros((M * R - X_i.shape[0], X_i.shape[1])), X_i))
            )
        else:
            X.append(X_i[-M * R :])

    # Aggregation
    X = np.stack(X)
    M_ = np.stack(M_)
    Y_pre = np.array(Y_pre)
    Y_post = np.array(Y_post)
    A = np.array(A)
    T = np.transpose(
        np.tile(np.arange(-M * R, 0), (N_SUBJECTS, X.shape[2], 1)), (0, 2, 1)
    )

    # Scaling
    X_to_scale = X.reshape((-1, X.shape[2]))  # (N, T, D) -> (N * T, D)
    X_scaled = scale(X_to_scale, axis=0)
    X = X_scaled.reshape(X.shape)

    Y = np.concatenate((Y_pre, Y_post), axis=1)
    Y_to_scale = Y.reshape((-1, 1))  # (N, M) + (N, H) -> (N * T, 1)
    Y_scaled = scale(Y_to_scale, axis=0)
    Y = Y_scaled.reshape(Y.shape)
    Y_pre, Y_post = Y[:, :-H], Y[:, -H:]

    A = scale(A, axis=0)  # [N, K]

    return X, M_, Y_pre, Y_post, A, T


def generate_and_write_data():
    X, M_, Y_pre, Y_post, A, T = load_data()
    joblib.dump(
        (X, M_, Y_pre, Y_post, A, T),
        os.path.join(os.getcwd(), "data/diabetes/processed.joblib"),
    )

    N = N_SUBJECTS
    D = X.shape[2]
    K = A.shape[1]
    C = 4
    joblib.dump(
        (N, M, H, R, D, K, C, X, M_, Y_pre, Y_post, A, T),
        os.path.join(os.getcwd(), "data/diabetes/hp_search.joblib"),
    )


def generate_data(return_raw=True):
    N_train = round(N_SUBJECTS * 0.8)
    N_test = round(N_SUBJECTS * 0.2)
    X, M_, Y_pre, Y_post, A, T = joblib.load(
        os.path.join(os.getcwd(), "data/diabetes/processed.joblib")
    )
    X_train, X_test = X[:N_train], X[N_train:]
    M_train, M_test = M_[:N_train], M_[N_train:]
    Y_pre_train, Y_pre_test = Y_pre[:N_train], Y_pre[N_train:]
    Y_post_train, Y_post_test = Y_post[:N_train], Y_post[N_train:]
    A_train, A_test = A[:N_train], A[N_train:]
    T_train, T_test = T[:N_train], T[N_train:]
    all_data = (
        N_SUBJECTS,
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
    return N_SUBJECTS, train_data, test_data
