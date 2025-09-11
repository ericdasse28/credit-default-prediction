"""Data preprocessing Transformers and pipelines."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class DropMissingRows(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.dropna(subset=self.columns)


def build_preprocessing_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            # Handle missing values
            ("drop_missing_loan_int_rates", DropMissingRows(columns=["loan_int_rate"])),
            ("person_emp_length_imputer", SimpleImputer(strategy="median")),
        ]
    )
    return pipeline
