"""Model training script."""

from __future__ import annotations

import os

import click
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from credit_default_prediction import params
from credit_default_prediction.dataset import LoanApplications
from credit_default_prediction.hyperparams import HyperParams
from credit_default_prediction.preprocessing import build_infered_transformers


def train(X: pd.DataFrame, y: pd.Series, hyper_parameters: HyperParams):
    loan_default_classifier = xgb.XGBClassifier(**hyper_parameters.to_dict())
    infered_transformers = build_infered_transformers(
        numeric_features=[
            "person_income",
            "person_emp_length",
            "person_age",
            "loan_percent_income",
            "loan_int_rate",
            "loan_amnt",
        ],
        categorical_features=["loan_grade", "loan_intent", "person_home_ownership"],
    )

    training_pipeline = Pipeline(
        steps=[
            ("infered_transformers", infered_transformers),
            ("classifier", loan_default_classifier),
        ]
    )
    training_pipeline.fit(X, y)

    return training_pipeline


def save_model_artifact(model, model_path):
    joblib.dump(model, model_path)


@click.command()
@click.option("--train-dataset-path", help="Path to the training dataset.")
@click.option("--model-path", help="Path where the model will be saved after training.")
def cli(train_dataset_path: os.PathLike, model_path: os.PathLike):
    train_dataset = LoanApplications.from_path(
        train_dataset_path,
        columns=params.get_important_features(),
    )

    hyperparameters = HyperParams.from_config()
    model = train(train_dataset.X, train_dataset.y, hyperparameters)

    save_model_artifact(model, model_path)
