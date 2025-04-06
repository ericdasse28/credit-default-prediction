"""Model training script."""

from dataclasses import asdict, dataclass

import joblib
import pandas as pd
import xgboost as xgb


@dataclass
class HyperParams:
    learning_rate: float
    max_depth: 4
    min_child_wieght: 1


def train(X: pd.Series, y: pd.Series, hyper_parameters: HyperParams):
    hyper_parameters_dict = asdict(hyper_parameters)
    loan_default_classifier = xgb.XGBClassifier(**hyper_parameters_dict)
    loan_default_classifier.fit(X, y)

    return loan_default_classifier


def save_model_artifact(model, model_path):
    joblib.dump(model, model_path)
