"""Data preprocessing Transformers and pipelines."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from credit_default_prediction.preserve_df import PreserveDF


class DropMissingRows(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.dropna(subset=self.columns)


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Wrapper around SimpleImputer that preserves
    DataFrame with column names.
    """

    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
        self.columns = None

    def fit(self, X: pd.DataFrame, y=None):
        self.columns = X.columns
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        X_t = self.imputer.transform(X)
        return pd.DataFrame(X_t, columns=self.columns, index=X.index)


class DropOutliersRows(BaseEstimator, TransformerMixin):
    """Clip employment_length to a maximum value (default=60)."""

    def __init__(self, column: str, max_value: float = 60):
        self.column = column
        self.max_value = max_value

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X[X[self.column] <= self.max_value]


def build_preprocessing_pipeline(categorical_features: list[str] = None) -> Pipeline:
    if not categorical_features:
        categorical_features = []

    emp_length_transformer = ColumnTransformer(
        transformers=[
            (
                "emp",
                Pipeline(
                    [
                        ("imputer", DataFrameImputer(strategy="median")),
                    ]
                ),
                ["person_emp_length"],
            )
        ],
        remainder="passthrough",  # Keep all other columns
    )
    cat_features_transformer = ColumnTransformer(
        transformers=[
            (
                "encoder",
                Pipeline([("encoder", OneHotEncoder())]),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    preprocessor = Pipeline(
        steps=[
            ("drop_missing_loan_int_rates", DropMissingRows(columns=["loan_int_rate"])),
            ("emp_length_imputer", PreserveDF(emp_length_transformer)),
            (
                "emp_length_outliers",
                DropOutliersRows(column="person_emp_length", max_value=60),
            ),
            ("cat", cat_features_transformer),
        ]
    )

    return preprocessor
