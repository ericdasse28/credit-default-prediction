"""Model training script."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from credit_default_prediction.feature_engineering import FeatureEngineer
from credit_default_prediction.preprocessing import build_preprocessing_pipeline
from credit_default_prediction.preserve_df import PreserveDF


@dataclass
class HyperParams:
    learning_rate: float
    max_depth: float = 4
    min_child_weight: float = 1

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, hyper_params: dict) -> HyperParams:
        return cls(**hyper_params)


def train(X: pd.DataFrame, y: pd.Series, hyper_parameters: HyperParams):
    loan_default_classifier = xgb.XGBClassifier(**hyper_parameters.to_dict())
    categorical_features = ["loan_grade", "loan_intent", "person_home_ownership"]
    preprocessing_pipeline = build_preprocessing_pipeline(
        categorical_features=categorical_features
    )

    training_pipeline = Pipeline(
        steps=[
            ("data_preprocessing", PreserveDF(preprocessing_pipeline)),
            ("feature_engineering", FeatureEngineer()),
            ("classifier", loan_default_classifier),
        ]
    )
    training_pipeline.fit(X, y)

    return training_pipeline


def save_model_artifact(model, model_path):
    joblib.dump(model, model_path)
