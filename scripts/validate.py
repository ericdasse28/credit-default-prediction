import argparse

import joblib
from sklearn.model_selection import KFold, cross_val_score

from credit_default_prediction.dataset import get_features_and_labels_from_path
from credit_default_prediction.metrics import save_model_metrics


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-path")
    parser.add_argument("--model-path")

    return parser.parse_args()


def main():
    args = _get_arguments()
    model = joblib.load(args.model_path)
    X_train, y_train = get_features_and_labels_from_path(
        args.train_dataset_path,
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    avg_accuracy = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="accuracy",
    ).mean()
    avg_precision = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="precision",
    ).mean()
    avg_recall = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="recall",
    ).mean()
    save_model_metrics(
        {
            "avg_accuracy": avg_accuracy,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
        },
        phase="cross_validation",
    )
