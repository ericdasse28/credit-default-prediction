import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from credit_default_prediction.data_preprocessing import (
    handle_features_types,
    handle_missing_values,
    handle_outliers,
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


def test_handle_features_types_make_cb_person_default_on_file_a_boolean_column():  # noqa
    """Given a dataframe containing loan applications data,
    When applying `handle_features_types`,
    Then the column `cb_person_default_on_file` turns into
    an integer column."""

    CB_DEFAULT_ON_FILE_COL = "cb_person_default_on_file"
    original_dataframe = pd.DataFrame(
        {
            "person_age": [22, 50, 21],
            CB_DEFAULT_ON_FILE_COL: ["Y", "N", "Y"],
        }
    )

    actual_dataframe = handle_features_types(
        original_dataframe,
    )

    expected_dataframe = pd.DataFrame(
        {
            "person_age": [22, 50, 21],
            CB_DEFAULT_ON_FILE_COL: [1, 0, 1],
        }
    )
    assert is_integer_dtype(actual_dataframe[CB_DEFAULT_ON_FILE_COL])
    pd.testing.assert_frame_equal(expected_dataframe, actual_dataframe)
