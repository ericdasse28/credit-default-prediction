import numpy as np
import pandas as pd

from credit_default_prediction.preprocess_data import (
    handle_missing_values,
    handle_outliers,
    preprocess_data,
)


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


def test_preprocess_pipeline_executes_steps_in_the_right_order(mocker):
    """Given a dataframe containing loan applications data,
    When applying data preprocessing function to it,
    Then the function should:
        1. Handle missing values
        2. Handle outliers
    In that order."""

    # Arrange
    sample_loan_data = pd.DataFrame(
        {
            "loan_int_rate": [11.84, np.nan, 12.5, 7.14, np.nan],
            "person_emp_length": [3, 0, 70, 60, 120],
        }
    )
    data_after_missing_values = sample_loan_data.copy()
    data_after_outlier_treatment = sample_loan_data.copy()
    # Mock
    mock_handle_missing_values = mocker.patch(
        "credit_default_prediction.preprocess_data.handle_missing_values",
        return_value=data_after_missing_values,
    )
    mock_handle_outliers = mocker.patch(
        "credit_default_prediction.preprocess_data.handle_outliers",
        return_value=data_after_outlier_treatment,
    )

    # Act
    clean_loan_data = preprocess_data(sample_loan_data)

    # Assert
    mock_handle_missing_values.assert_called_once_with(sample_loan_data)
    mock_handle_outliers.assert_called_once_with(data_after_missing_values)
    pd.testing.assert_frame_equal(
        clean_loan_data,
        data_after_outlier_treatment,
    )
