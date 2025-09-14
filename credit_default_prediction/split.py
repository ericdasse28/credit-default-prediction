from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default_prediction import params
from credit_default_prediction.dataset import LoanApplications


@dataclass
class SplitParams:
    test_size: float
    random_state: int

    @classmethod
    def from_config(cls) -> SplitParams:
        pipeline_params = params.load_stage_params("split")
        return SplitParams(
            test_size=pipeline_params["test_size"],
            random_state=pipeline_params["random_state"],
        )


def split_data(
    loan_data: pd.DataFrame, split_params: SplitParams
) -> tuple[LoanApplications, LoanApplications]:
    """Splits loan data into training and test features and labels (X_train, X_test, y_train, y_test).

    Args:
        loan_data (pd.DataFrame): Loan applications data containing features and target.
        split_params (SplitParams): Parameters required to appropriately split the data.

    Returns:
        tuple[Dataset, Dataset]: Split data.
    """

    loan_dataset = LoanApplications.from_dataframe(loan_data)
    X_train, X_test, y_train, y_test = train_test_split(
        loan_dataset.X,
        loan_dataset.y,
        test_size=split_params.test_size,
        random_state=split_params.random_state,
    )

    training_data = LoanApplications(X=X_train, y=y_train)
    test_data = LoanApplications(X=X_test, y=y_test)
    return training_data, test_data


def split_data_from_path(
    raw_data_path: os.PathLike,
    split_data_dir: os.PathLike,
    split_params: SplitParams,
):
    """Reads raw loan applications from `raw_data_path`
    and splits them into training and test datasets.

    Args:
        raw_data_path (os.PathLike): Path to raw loan applications.
        split_data_dir (os.PathLike): Directory where training and test datasets CSV files are to be saved.
        split_params (SplitParams): Parameters required to appropriately split the data.
    """

    loan_applications = pd.read_csv(raw_data_path)
    train_path = Path(split_data_dir) / "train.csv"
    test_path = Path(split_data_dir) / "test.csv"

    training_dataset, test_dataset = split_data(loan_applications, split_params)

    training_dataset.save_to_csv(train_path)
    test_dataset.save_to_csv(test_path)


@click.command()
@click.option("--raw-data-path", help="Path to raw data.")
@click.option(
    "--split-data-dir", help="Path where training and test CSV datasets will be saved."
)
def cli(raw_data_path: os.PathLike, split_data_dir: os.PathLike):
    split_params = SplitParams.from_config()

    split_data_from_path(
        raw_data_path=raw_data_path,
        split_data_dir=split_data_dir,
        split_params=split_params,
    )
