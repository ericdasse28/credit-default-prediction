import argparse
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default_prediction.dataset import get_features_and_labels
from credit_default_prediction.tools.params import load_stage_params


@dataclass
class SplitParams:
    test_size: float
    random_state: int


def split_data(loan_data: pd.DataFrame, split_params: SplitParams):
    X, y = get_features_and_labels(loan_data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_params.test_size,
        random_state=split_params.random_state,
    )
    return X_train, X_test, y_train, y_test


def load_split_params() -> SplitParams:
    pipeline_params = load_stage_params("split")
    return SplitParams(
        test_size=pipeline_params["test_size"],
        random_state=pipeline_params["random_state"],
    )


def save_loan_data(X, y, save_path):
    loan_data = pd.concat([X, y], axis=1)
    loan_data.to_csv(save_path, index=False)


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-data-path")
    parser.add_argument("--train-path")
    parser.add_argument("--test-path")

    return parser.parse_args()


def main():
    args = _get_arguments()

    loan_data = pd.read_csv(args.preprocessed_data_path)
    split_params = load_split_params()
    X_train, X_test, y_train, y_test = split_data(loan_data, split_params)

    save_loan_data(X_train, y_train, args.train_path)
    save_loan_data(X_test, y_test, args.test_path)
