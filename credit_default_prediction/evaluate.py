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
    with Live(resume=True) as live:
        for metric in metrics.keys():
            live.log_metric(f"test/{metric}", metrics[metric])


def log_confusion_matrix(model, X: pd.DataFrame, y: pd.DataFrame, live: Live):
    predictions = model.predict(X)
    preds_df = pd.DataFrame()
    preds_df["actual"] = y.values
    preds_df["predicted"] = predictions

    live.log_plot(
        "confusion_matrix",
        preds_df,
        x="actual",
        y="predicted",
        template="confusion",
        x_label="Actual labels",
        y_label="Predicted labels",
    )


def log_roc_curve(model, X: pd.DataFrame, y: pd.DataFrame, live: Live):
    y_score = model.predict_proba(X)[:, 1].astype(float)
    live.log_sklearn_plot("roc", y, y_score)


def log_feature_importance(model, live: Live):
    feature_importance = pd.Series(model.get_booster().get_score())
    feature_importance = feature_importance.reset_index()
    feature_importance = feature_importance.rename(
        columns={"index": "feature_name", 0: "feature_importance"}
    )

    live.log_plot(
        "feature_importance",
        feature_importance,
        x="feature_importance",
        y="feature_name",
        template="bar_horizontal",
    )


def log_plots(model, X: pd.DataFrame, y: pd.DataFrame):
    with Live(resume=True) as live:
        log_confusion_matrix(model, X, y, live)
        log_roc_curve(model, X, y, live)
        log_feature_importance(model, live)


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
    log_plots(model, X_test, y_test)
