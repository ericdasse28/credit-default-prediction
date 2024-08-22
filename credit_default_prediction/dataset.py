"""Dataset-related functions."""

import os

import pandas as pd


def get_features_and_labels(loan_data: pd.DataFrame) -> tuple[
    pd.Series,
    pd.Series,
]:
    X = loan_data.drop("loan_status", axis=1)
    y = loan_data["loan_status"]

    return X, y


def read_features_and_labels(dataset_path: os.PathLike) -> tuple[
    pd.Series,
    pd.Series,
]:
    loan_data = pd.read_csv(dataset_path)
    X, y = get_features_and_labels(loan_data)

    return X, y
