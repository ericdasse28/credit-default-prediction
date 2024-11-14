import numpy as np
import pandas as pd

from credit_default_prediction.preprocess import (
    handle_missing_values,
    handle_outliers,
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


def test_handle_missing_values_should_drop_rows_with_missing_loan_interests():
    """Given a dataframe containing loan applications data,
    When handling its missing values,
    Then it should drop rows with missing loan interests."""

    loan_data = pd.DataFrame(
        {
            "loan_int_rate": [11.84, np.nan, 12.5, 7.14, np.nan],
            "person_age": [20, 50, 18, 65, 19],
            "person_emp_length": [2, 15, 1.3, 20, 3],
        }
    )

    actual_clean_loan_data = handle_missing_values(loan_data)

    expected_clean_loan_data = pd.DataFrame(
        {
            "loan_int_rate": [11.84, 12.5, 7.14],
            "person_age": [20, 18, 65],
            "person_emp_length": [2, 1.3, 20],
        },
        index=[0, 2, 3],
    )
    pd.testing.assert_frame_equal(
        actual_clean_loan_data,
        expected_clean_loan_data,
    )


def test_handle_missing_values_should_impute_missing_emp_length():
    """Given a dataframe containing loan applications data,
    When handling missing values,
    Then it should impute missing employment lengths with the median
    employment length."""

    loan_data = pd.DataFrame(
        {
            "person_age": [40, 35, 50, 40, 19],
            "loan_int_rate": [11.5, 8.3, 4.5, 6.9, 7.8],
            "person_emp_length": [np.nan, 12, 13, 25, 3],
        }
    )

    actual_clean_loan_data = handle_missing_values(loan_data)

    median_emp_length = loan_data["person_emp_length"].median()
    expected_clean_loan_data = pd.DataFrame(
        {
            "person_age": [40, 35, 50, 40, 19],
            "loan_int_rate": [11.5, 8.3, 4.5, 6.9, 7.8],
            "person_emp_length": [median_emp_length, 12, 13, 25, 3],
        }
    )
    pd.testing.assert_frame_equal(
        actual_clean_loan_data,
        expected_clean_loan_data,
    )


def test_handle_outliers_should_drop_rows_with_outlier_emp_length():
    """Given a dataframe containing loan applications data,
    When handling outliers,
    Then it should drop rows with outlier employment lengths."""

    loan_data = pd.DataFrame(
        {
            "person_emp_length": [3, 0, 70, 60, 120],
            "loan_status": [1, 1, 0, 0, 1],
        }
    )

    actual_clean_loan_data = handle_outliers(loan_data)

    expected_clean_loan_data = pd.DataFrame(
        {"person_emp_length": [3, 0, 60], "loan_status": [1, 1, 0]},
        index=[0, 1, 3],
    )
    pd.testing.assert_frame_equal(
        actual_clean_loan_data,
        expected_clean_loan_data,
    )
