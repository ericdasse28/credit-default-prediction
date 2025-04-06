from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default_prediction.dataset import get_features_and_labels


@dataclass
class SplitParams:
    test_size: float
    random_state: int


def split_data(
    loan_data: pd.DataFrame, split_params: SplitParams
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and targets training and test sets.

    Args:
        loan_data (pd.DataFrame): Loan applications data containing features and target.
        split_params (SplitParams): Parameters required to appropriately split the data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Split data.
    """

    X, y = get_features_and_labels(loan_data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_params.test_size,
        random_state=split_params.random_state,
    )
    return X_train, X_test, y_train, y_test
