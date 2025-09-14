"""Model training script."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import click
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from credit_default_prediction import params
from credit_default_prediction.dataset import LoanApplications
from credit_default_prediction.preprocessing import build_infered_transformers


@dataclass
class HyperParams:
    learning_rate: float
    max_depth: float = 4
    min_child_weight: float = 1

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, hyper_params: dict[str, float]) -> HyperParams:
        return cls(**hyper_params)

    @classmethod
    def from_config(cls) -> HyperParams:
        hyperparams_config = params.get_hyperparameters()
        return cls.from_dict(hyperparams_config)


def train(X: pd.DataFrame, y: pd.Series, hyper_parameters: HyperParams):
    # Build infered transformers
    all_numeric_features = X.select_dtypes(include=np.number).columns.to_list()
    all_categorical_features = X.select_dtypes(include=object).columns.to_list()
    infered_transformers = build_infered_transformers(
        numeric_features=all_numeric_features,
        categorical_features=all_categorical_features,
    )
    # Classifier
    loan_default_classifier = xgb.XGBClassifier(**hyper_parameters.to_dict())

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
