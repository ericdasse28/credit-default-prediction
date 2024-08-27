"""Model training script."""

import argparse

import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from credit_default_prediction.dataset import read_features_and_labels


def train(X: pd.Series, y: pd.Series):
    loan_default_classifier = KNeighborsClassifier(n_neighbors=10)
    loan_default_classifier.fit(X, y)

    return loan_default_classifier


def save_model_artifact(model, model_path):
    joblib.dump(model, model_path)


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-path")
    parser.add_argument("--model-path")

    return parser.parse_args()


def main():
    args = _get_arguments()
    X_train, y_train = read_features_and_labels(args.train_dataset_path)

    model = train(X_train.values, y_train.values)
    save_model_artifact(model, args.model_path)
