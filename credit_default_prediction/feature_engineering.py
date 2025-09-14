import os

import click
import numpy as np
import pandas as pd


def log_transform_large_features(loan_data: pd.DataFrame) -> pd.DataFrame:
    """Applies log transformation to large features."""
    LARGE_FEATURES = ["person_income", "loan_amnt"]

    clean_loan_data = loan_data.copy()
    clean_loan_data[LARGE_FEATURES] = loan_data[LARGE_FEATURES].apply(
        lambda row: np.log1p(row),  # We add 1 to log argument to avoid log(0) # noqa
    )

    return clean_loan_data


def engineer_features(clean_loan_data: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on input clean loan
    applications dataframe."""

    # Log tranform large features
    feature_engineered_data = log_transform_large_features(
        clean_loan_data,
    )

    return feature_engineered_data


@click.command(
    help="Performs feature engineering on loan applications data that have preprocessed beforehand. This operation transforms the data to help the model learn better."
)
@click.option(
    "--preprocessed-data-path",
    help="Path to the preprocessed loan applications CSV file.",
)
@click.option(
    "--feature-store-path",
    help="Path where the feature engineered loan applications CSV file will be saved.",
)
def cli(preprocessed_data_path: os.PathLike, feature_store_path: os.PathLike):
    preprocessed_data = pd.read_csv(preprocessed_data_path)
    preprocessed_data = engineer_features(preprocessed_data)
    # Save feature engineered loan applications
    preprocessed_data.to_csv(feature_store_path, index=False)
