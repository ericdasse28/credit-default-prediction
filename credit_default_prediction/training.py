"""Model training script."""

from __future__ import annotations

import os

import click
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dvclive.live import Live
from sklearn.pipeline import Pipeline

from credit_default_prediction import params
from credit_default_prediction.dataset import LoanApplications
from credit_default_prediction.experiment_tracking.dvc_experiment_tracker import (
    DVCExperimentTracker,
)
from credit_default_prediction.hyper_params import HyperParams
from credit_default_prediction.preprocessing import build_infered_transformers


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

    with Live() as live:
        experiment_tracker = DVCExperimentTracker(live)

        hyperparameters = HyperParams.from_config()
        model = train(train_dataset.X, train_dataset.y, hyperparameters)

        experiment_tracker.log_params(hyperparameters)

    save_model_artifact(model, model_path)
