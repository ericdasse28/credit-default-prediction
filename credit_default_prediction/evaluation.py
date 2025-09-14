"""Evaluation script."""

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (  # noqa
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from dvclive import Live


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "ROC_AUC": roc_auc_score(y, model.predict_proba(X)[:, 1]),
    }


def log_confusion_matrix(model: Pipeline, X: pd.DataFrame, y: pd.Series, live: Live):
    predictions = model.predict(X)
    preds_df = pd.DataFrame()
    preds_df["actual"] = y.values
    preds_df["predicted"] = predictions

    live.log_plot(
        "confusion_matrix",
        preds_df,
        x="actual",
        y="predicted",
        template="confusion",
        x_label="Actual labels",
        y_label="Predicted labels",
    )


def log_roc_curve(model: Pipeline, X: pd.DataFrame, y: pd.Series, live: Live):
    y_score = model.predict_proba(X)[:, 1].astype(float)
    y_true = y.to_numpy()
    live.log_sklearn_plot("roc", y_true, y_score)


def generate_feature_importance_data(
    model: xgb.XGBClassifier, feature_names: list[str]
) -> pd.DataFrame:
    booster = model.get_booster()
    booster.feature_names = feature_names
    importance_per_feature = booster.get_score()

    feature_importance = pd.DataFrame(
        importance_per_feature.items(),
        columns=["feature_name", "feature_importance"],
    )

    return feature_importance


def log_feature_importance_plot(model: Pipeline, live: Live):
    classifier = model.named_steps["classifier"]
    feature_names = list(
        model.named_steps["infered_transformers"].get_feature_names_out()
    )

    feature_importance = generate_feature_importance_data(classifier, feature_names)

    live.log_plot(
        "feature_importance",
        feature_importance,
        x="feature_importance",
        y="feature_name",
        template="bar_horizontal",
    )


def log_plots(model: Pipeline, X: pd.DataFrame, y: pd.Series):
    with Live(resume=True) as live:
        log_confusion_matrix(model, X, y, live)
        log_roc_curve(model, X, y, live)
        log_feature_importance_plot(model, live)
