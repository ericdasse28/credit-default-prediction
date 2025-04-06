import numpy as np
import pandas as pd

from credit_default_prediction.tools import params


def get_important_features() -> list[str]:
    preprocess_params = params.load_stage_params("feature_engineering")
    return preprocess_params["important_columns"]


def log_transform_large_features(loan_data: pd.DataFrame) -> pd.DataFrame:
    """Applies log transformation to large features."""
    LARGE_FEATURES = ["person_income", "loan_amnt"]

    clean_loan_data = loan_data.copy()
    clean_loan_data[LARGE_FEATURES] = loan_data[LARGE_FEATURES].apply(
        lambda row: np.log(row + 1),  # We add 1 to log argument to avoid log(0) # noqa
    )

    return clean_loan_data


def engineer_features(clean_loan_data: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on input clean loan
    applications dataframe."""

    # Feature selection
    feature_engineered_data = clean_loan_data[get_important_features()]
    # One-hot encoding
    feature_engineered_data = pd.get_dummies(feature_engineered_data)
    # Log tranform large features
    feature_engineered_data = log_transform_large_features(
        feature_engineered_data,
    )

    return feature_engineered_data
