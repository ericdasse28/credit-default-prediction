import pandas as pd
import pytest

from credit_default_prediction.split import SplitParams, split_data


def test_split_data():
    """Given loan applications dataframe and split parameters,
    When we split the loan applications dataframe using said parameters,
    Then we obtain training and test data."""

    dummy_data = pd.DataFrame(
        {
            "person_age": [22, 25, 40],
            "person_income": [35000, 45000, 52000],
            "loan_intent": ["PERSONAL", "MEDICAL", "VENTURE"],
            "loan_status": [0, 1, 1],
        }
    )
    dummy_parameters = SplitParams(test_size=0.2, random_state=3)

    X_train, X_test, y_train, y_test = split_data(dummy_data, dummy_parameters)

    assert len(X_train) == 2
    assert len(y_train) == 2
    assert len(X_test) == 1
    assert len(y_test) == 1


def test_split_data_missing_loan_status():
    """Given loan applications dataframe with a missing 'loan_status' column
      and split parameters,
    When we split the loan applications dataframe using said parameters,
    Then a KeyError exception about the missing 'loan_status' column is raised."""

    dummy_data = pd.DataFrame(
        {
            "person_age": [22, 25, 40],
            "person_income": [35000, 45000, 52000],
            "loan_intent": ["PERSONAL", "MEDICAL", "VENTURE"],
        }
    )
    dummy_parameters = SplitParams(test_size=0.2, random_state=3)

    with pytest.raises(KeyError) as e_info:
        X_train, X_test, y_train, y_test = split_data(dummy_data, dummy_parameters)

    assert "loan_status" in str(e_info.value)
