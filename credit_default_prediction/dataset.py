"""Dataset-related functions."""

import os

import pandas as pd


def get_features_and_labels(loan_data: pd.DataFrame) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    X = loan_data.drop("loan_status", axis=1)
    y = loan_data["loan_status"]

    return X, y


def get_features_and_labels_from_path(dataset_path: os.PathLike) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    loan_data = pd.read_csv(dataset_path)
    X, y = get_features_and_labels(loan_data)

    return X, y
