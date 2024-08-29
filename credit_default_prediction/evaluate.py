"""Evaluation script."""

import argparse

import joblib
import pandas as pd
from sklearn.metrics import (  # noqa
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_default_prediction.dataset import read_features_and_labels
from dvclive import Live


def evaluate(model, X: pd.Series, y: pd.Series) -> dict:
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "ROC_AUC": roc_auc_score(y, model.predict_proba(X)[:, 1]),
    }


def save_model_metrics(metrics: dict):
    with Live() as live:
        for metric in metrics.keys():
            live.log_metric(f"test/{metric}", metrics[metric])


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--test-dataset-path")

    return parser.parse_args()


def main():

    args = _get_arguments()
    model = joblib.load(args.model_path)
    X_test, y_test = read_features_and_labels(args.test_dataset_path)

    metrics = evaluate(model, X_test.values, y_test.values)
    save_model_metrics(metrics)
