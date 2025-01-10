import argparse

import numpy as np
import pandas as pd

from credit_default_prediction.tools import params


def get_important_features() -> list[str]:
    preprocess_params = params.load_stage_params("feature_engineering")
    return preprocess_params["important_columns"]


def engineer_features(clean_loan_data: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on input clean loan
    applications dataframe."""

    # Feature selection
    feature_engineered_data = clean_loan_data[get_important_features()]
    # One-hot encoding
    feature_engineered_data = pd.get_dummies(feature_engineered_data)

    return feature_engineered_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-data-path")
    parser.add_argument("--feature-store-path")
    args = parser.parse_args()

    preprocessed_data = pd.read_csv(args.preprocessed_data_path)
    preprocessed_data = engineer_features(preprocessed_data)
    # Save feature engineered features
    preprocessed_data.to_csv(args.feature_store_path, index=False)


def log_transform_large_features(loan_data: pd.DataFrame) -> pd.DataFrame:
    """Applies log transformation to large features."""
    LARGE_FEATURES = ["person_income", "loan_amnt"]

    clean_loan_data = loan_data.copy()
    clean_loan_data[LARGE_FEATURES] = loan_data[LARGE_FEATURES].apply(
        lambda row: np.log(row + 1),  # We add 1 to log argument to avoid log(0) # noqa
    )

    return clean_loan_data
