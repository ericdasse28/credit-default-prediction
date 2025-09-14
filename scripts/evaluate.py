import argparse

import joblib

from credit_default_prediction import inference, params
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
    test_dataset = LoanApplications.from_path(
        args.test_dataset_path,
        columns=params.get_important_features(),
    )
    # Prepare test dataset
    prepped_test_data = inference.rule_based_preparation(test_dataset.data)
    test_dataset = LoanApplications.from_dataframe(prepped_test_data)

    # Evaluate trained model on prepped
    trained_model = joblib.load(args.model_path)
    X_test, y_test = test_dataset.X, test_dataset.y
    metrics = evaluate(trained_model, X_test, y_test)

    # Save metrics and plots
    save_model_metrics(metrics)
    log_plots(trained_model, X_test, y_test)
