import argparse

import numpy as np

from credit_default_prediction.dataset import collect_loan_dataset_from_path
from credit_default_prediction.tools import params
from credit_default_prediction.training import save_model_artifact, train


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-path")
    parser.add_argument("--model-path")

    return parser.parse_args()


def main():
    args = _get_arguments()
    X_train, y_train = collect_loan_dataset_from_path(
        args.train_dataset_path,
    )

    hyperparameters = params.get_hyperparameters()
    model = train(X_train, np.ravel(y_train), hyperparameters)
    save_model_artifact(model, args.model_path)
