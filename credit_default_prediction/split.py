from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default_prediction.dataset import Dataset, get_features_and_labels


@dataclass
class SplitParams:
    test_size: float
    random_state: int


def split_data(
    loan_data: pd.DataFrame, split_params: SplitParams
) -> tuple[Dataset, Dataset]:
    """Splits loan data into training and test features and labels (X_train, X_test, y_train, y_test).

    Args:
        loan_data (pd.DataFrame): Loan applications data containing features and target.
        split_params (SplitParams): Parameters required to appropriately split the data.

    Returns:
        tuple[Dataset, Dataset]: Split data.
    """

    loan_dataset = get_features_and_labels(loan_data)
    X_train, X_test, y_train, y_test = train_test_split(
        loan_dataset.X,
        loan_dataset.y,
        test_size=split_params.test_size,
        random_state=split_params.random_state,
    )

    training_data = Dataset(X=X_train, y=y_train)
    test_data = Dataset(X=X_test, y=y_test)
    return training_data, test_data
