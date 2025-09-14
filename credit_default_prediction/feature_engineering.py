import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return engineer_features(X)
