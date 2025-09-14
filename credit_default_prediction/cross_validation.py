import os

import click
import joblib
from sklearn.model_selection import KFold, cross_val_score

from credit_default_prediction.dataset import LoanApplications
from credit_default_prediction.metrics import save_model_metrics


@click.command(
    help="Performs cross-validation of the trained model on the training data."
)
@click.option(
    "--train-dataset-path", help="Path to the CSV feature-engineered training data."
)
@click.option("--model-path", help="Path to the trained model.")
def cli(train_dataset_path: os.PathLike, model_path: os.PathLike):
    trained_model = joblib.load(model_path)
    train_dataset = LoanApplications.from_path(
        train_dataset_path,
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    avg_accuracy = cross_val_score(
        trained_model,
        train_dataset.X,
        train_dataset.y,
        cv=kf,
        scoring="accuracy",
    ).mean()
    avg_precision = cross_val_score(
        trained_model,
        train_dataset.X,
        train_dataset.y,
        cv=kf,
        scoring="precision",
    ).mean()
    avg_recall = cross_val_score(
        trained_model,
        train_dataset.X,
        train_dataset.y,
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
