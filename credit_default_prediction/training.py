"""Model training script."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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

    training_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("encoder", OneHotEncoder()),
            ("classifier", loan_default_classifier),
        ]
    )
    training_pipeline.fit(X, y)

    return training_pipeline


def save_model_artifact(model, model_path):
    joblib.dump(model, model_path)
