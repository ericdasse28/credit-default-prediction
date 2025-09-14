"""Model training script."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from credit_default_prediction.preprocessing import build_infered_transformers


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
    infered_transformers = build_infered_transformers(
        numeric_features=[
            "person_income",
            "person_emp_length",
            "person_age",
            "loan_percent_income",
            "loan_int_rate",
            "loan_amnt",
        ],
        categorical_features=["loan_grade", "loan_intent", "person_home_ownership"],
    )

    training_pipeline = Pipeline(
        steps=[
            ("infered_transformers", infered_transformers),
            ("classifier", loan_default_classifier),
        ]
    )
    training_pipeline.fit(X, y)

    return training_pipeline


def save_model_artifact(model, model_path):
    joblib.dump(model, model_path)
