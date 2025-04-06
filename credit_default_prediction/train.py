"""Model training script."""

import joblib
import pandas as pd
import xgboost as xgb


def train(X: pd.Series, y: pd.Series, hyper_parameters: dict):
    loan_default_classifier = xgb.XGBClassifier(**hyper_parameters)
    loan_default_classifier.fit(X, y)

    return loan_default_classifier


def save_model_artifact(model, model_path):
    joblib.dump(model, model_path)
