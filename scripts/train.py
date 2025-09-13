import argparse

import numpy as np

from credit_default_prediction import params
from credit_default_prediction.dataset import LoanApplications
from credit_default_prediction.training import HyperParams, save_model_artifact, train


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-path")
    parser.add_argument("--model-path")

    return parser.parse_args()


def main():
    args = _get_arguments()
    train_dataset = LoanApplications.from_path(
        args.train_dataset_path,
        columns=params.get_important_features(),
    )

    hyperparameters = HyperParams.from_dict(params.get_hyperparameters())
    model = train(train_dataset.X, np.ravel(train_dataset.y), hyperparameters)

    save_model_artifact(model, args.model_path)
