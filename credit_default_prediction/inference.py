import pandas as pd

from credit_default_prediction.feature_engineering import engineer_features
from credit_default_prediction.preprocessing import rule_based_preprocessing


def rule_based_preparation(loan_data: pd.DataFrame) -> pd.DataFrame:
    prepped_data = rule_based_preprocessing(loan_data)
    prepped_data = engineer_features(prepped_data)

    return prepped_data
