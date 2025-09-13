import argparse

import joblib

from credit_default_prediction import params
from credit_default_prediction.dataset import LoanApplications
from credit_default_prediction.evaluation import evaluate, log_plots
from credit_default_prediction.metrics import save_model_metrics


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--test-dataset-path")

    return parser.parse_args()


def main():

    args = _get_arguments()
    model = joblib.load(args.model_path)
    test_dataset = LoanApplications.from_path(
        args.test_dataset_path,
        columns=params.get_important_features(),
    )
    X_test, y_test = test_dataset.X, test_dataset.y

    metrics = evaluate(model, X_test.values, y_test.values)
    save_model_metrics(metrics)
    log_plots(model, X_test, y_test)
