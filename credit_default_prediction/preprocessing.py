"""Data preprocessing Transformers and pipelines."""

import os

import click
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PERSON_EMP_LENGTH_MAX = 60


def rule_based_preprocessing(loan_data: pd.DataFrame) -> pd.DataFrame:
    # Loan interests are mandatory. Observations
    # without it are not helpful
    clean_loan_data = loan_data.dropna(subset="loan_int_rate")
    # Removing outlier employment lengths
    clean_loan_data = clean_loan_data[
        clean_loan_data["person_emp_length"] <= PERSON_EMP_LENGTH_MAX
    ]

    return clean_loan_data


def build_infered_transformers(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    num_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_transformer = Pipeline([("encoder", OneHotEncoder())])

    infered_transformers = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features),
        ]
    )
    return infered_transformers


@click.command(
    help="Performs deterministic data preprocessing steps. Data preprocessing cleans the data so they represent reality faithfully."
)
@click.option("--raw-data-path", help="Path to a CSV raw loan applications dataset.")
@click.option(
    "--preprocessed-data-path",
    help="Path where the preprocessed data CSV will be saved.",
)
def cli(raw_data_path: os.PathLike, preprocessed_data_path: os.PathLike):
    loan_data = pd.read_csv(raw_data_path)
    loan_data = rule_based_preprocessing(loan_data)
    loan_data.to_csv(preprocessed_data_path, index=False)
