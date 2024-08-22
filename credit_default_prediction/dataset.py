"""Dataset-related functions."""

import pandas as pd


def get_features_and_labels(loan_data: pd.DataFrame) -> tuple[
    pd.Series,
    pd.Series,
]:
    X = loan_data.drop("loan_status", axis=1)
    y = loan_data["loan_status"]

    return X, y
