"""Dataset-related functions."""

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series


def collect_loan_dataset(loan_data: pd.DataFrame) -> Dataset:
    X = loan_data.drop("loan_status", axis=1)
    y = loan_data["loan_status"]

    return Dataset(X=X, y=y)


def collect_loan_dataset_from_path(
    dataset_path: os.PathLike, columns: list[str] = None
) -> Dataset:
    loan_data = pd.read_csv(dataset_path)
    if columns:
        loan_data = loan_data[columns]

    loan_dataset = collect_loan_dataset(loan_data)

    return Dataset(X=loan_dataset.X, y=loan_dataset.y)
