import argparse

import joblib
import pandas as pd

from credit_default_prediction.data_preprocessing import preprocess_data
from credit_default_prediction.dataset import collect_loan_dataset
from credit_default_prediction.evaluation import evaluate, log_plots
from credit_default_prediction.feature_engineering import engineer_features
from credit_default_prediction.metrics import save_model_metrics


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--test-dataset-path")

    return parser.parse_args()


def main():

    args = _get_arguments()
    model = joblib.load(args.model_path)
    test_data = pd.read_csv(args.test_dataset_path)
    preprocessed_test_data = preprocess_data(test_data)
    preprocessed_test_data = engineer_features(preprocessed_test_data)
    X_test, y_test = collect_loan_dataset(preprocessed_test_data)

    metrics = evaluate(model, X_test.values, y_test.values)
    save_model_metrics(metrics)
    log_plots(model, X_test, y_test)
