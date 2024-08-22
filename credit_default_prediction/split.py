import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default_prediction.dataset import get_features_and_labels


def split_data(loan_data):
    X, y = get_features_and_labels(loan_data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )
    return X_train, X_test, y_train, y_test


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
    X_train, X_test, y_train, y_test = split_data(loan_data)

    save_loan_data(X_train, y_train, args.train_path)
    save_loan_data(X_test, y_test, args.test_path)
