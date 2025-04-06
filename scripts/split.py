import argparse

import pandas as pd

from credit_default_prediction.split import SplitParams, split_data
from credit_default_prediction.tools.params import load_stage_params


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path")
    parser.add_argument("--train-path")
    parser.add_argument("--test-path")

    return parser.parse_args()


def _load_split_params() -> SplitParams:
    pipeline_params = load_stage_params("split")
    return SplitParams(
        test_size=pipeline_params["test_size"],
        random_state=pipeline_params["random_state"],
    )


def _save_loan_data(X, y, save_path):
    loan_data = pd.concat([X, y], axis=1)
    loan_data.to_csv(save_path, index=False)


def main():
    args = _get_arguments()

    loan_data = pd.read_csv(args.raw_data_path)
    split_params = _load_split_params()
    X_train, X_test, y_train, y_test = split_data(loan_data, split_params)

    _save_loan_data(X_train, y_train, args.train_path)
    _save_loan_data(X_test, y_test, args.test_path)
