import numpy as np
import pandas as pd

from credit_default_prediction.preprocess import (
    remove_missing_loan_interests_rows,
    remove_outliers,
    replace_missing_emp_length,
    select_important_columns,
)


def test_remove_outliers():
    """Given credit loan dataframe,
    When removing outliers from it,
    Then there must be no loan observations
    with over 60 years of employment length."""

    loan_data = pd.DataFrame(
        {
            "person_emp_length": [3, 0, 70, 60, 120],
            "loan_status": [1, 1, 0, 0, 1],
        }
    )

    actual_clean_loan_data = remove_outliers(loan_data)

    expected_clean_loan_data = pd.DataFrame(
        {"person_emp_length": [3, 0, 60], "loan_status": [1, 1, 0]},
        index=[0, 1, 3],
    )
    pd.testing.assert_frame_equal(
        actual_clean_loan_data,
        expected_clean_loan_data,
    )


def test_remove_missing_loan_interests_rows():
    """Given credit loan data,
    When we remove the missing values,
    All rows with missing loan interest rates are gone."""

    loan_data = pd.DataFrame(
        {
            "loan_int_rate": [11.84, np.nan, 12.5, 7.14, np.nan],
            "person_age": [20, 50, 18, 65, 19],
        }
    )

    actual_clean_loan_data = remove_missing_loan_interests_rows(loan_data)

    expected_clean_loan_data = pd.DataFrame(
        {"loan_int_rate": [11.84, 12.5, 7.14], "person_age": [20, 18, 65]},
        index=[0, 2, 3],
    )
    pd.testing.assert_frame_equal(
        actual_clean_loan_data,
        expected_clean_loan_data,
    )


def test_replace_missing_emp_length():
    """Given credit loan data,
    When we replace missing employment length,
    Then all missing employment length values are replaced
    with the median employment length."""

    loan_data = pd.DataFrame(
        {
            "person_age": [40, 35, 50, 40, 19],
            "person_emp_length": [np.nan, 12, 13, 25, 3],
        }
    )

    actual_clean_loan_data = replace_missing_emp_length(loan_data)

    expected_clean_loan_data = pd.DataFrame(
        {
            "person_age": [40, 35, 50, 40, 19],
            "person_emp_length": [12.5, 12, 13, 25, 3],
        }
    )
    pd.testing.assert_frame_equal(
        actual_clean_loan_data,
        expected_clean_loan_data,
    )


def test_select_important_features():
    loan_data = pd.DataFrame(
        {
            "person_age": [40, 35, 50, 40, 19],
            "person_emp_length": [np.nan, 12, 13, 25, 3],
            "loan_intent": [
                "MEDICAL",
                "PERSONAL",
                "PERSONAL",
                "MEDICAL",
                "MEDICAL",
            ],
            "loan_grade": ["A", "A", "B", "G", "E"],
        }
    )
    important_features = ["person_age", "person_emp_length"]

    actual_loan_data = select_important_columns(loan_data, important_features)

    expected_loan_data = pd.DataFrame(
        {
            "person_age": [40, 35, 50, 40, 19],
            "person_emp_length": [np.nan, 12, 13, 25, 3],
        }
    )
    pd.testing.assert_frame_equal(actual_loan_data, expected_loan_data)
