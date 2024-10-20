"""Model training script."""

import argparse

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from credit_default_prediction.dataset import get_features_and_labels_from_path
from credit_default_prediction.tools import params


def train(X: pd.Series, y: pd.Series, hyper_parameters: dict):
    loan_default_classifier = xgb.XGBClassifier(**hyper_parameters)
    loan_default_classifier.fit(X, y)

    return loan_default_classifier


def save_model_artifact(model, model_path):
    joblib.dump(model, model_path)


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-path")
    parser.add_argument("--model-path")

    return parser.parse_args()


def main():
    args = _get_arguments()
    X_train, y_train = get_features_and_labels_from_path(
        args.train_dataset_path,
    )

    hyperparameters = params.get_hyperparameters()
    model = train(X_train, np.ravel(y_train), hyperparameters)
    save_model_artifact(model, args.model_path)
