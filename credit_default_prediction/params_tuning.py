"""Hyper-parameter tuning script."""

import argparse

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from credit_default_prediction.dataset import get_features_and_labels_from_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path")
    args = parser.parse_args()
    X_train, y_train = get_features_and_labels_from_path(args.dataset_path)

    params_to_test = {
        "learning_rate": np.arange(0, 1, 0.1),
        "max_depth": range(3, 10),
        "min_child_weight": range(1, 6),
    }

    gsearch = GridSearchCV(
        estimator=xgb.XGBClassifier(
            learning_rate=0.3,
            max_depth=5,
        ),
        param_grid=params_to_test,
        scoring="roc_auc",
    )
    gsearch.fit(X_train, y_train)
    print(gsearch.best_params_, gsearch.best_score_)
