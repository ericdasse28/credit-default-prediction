import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from pytest_mock import MockerFixture

from credit_default_prediction import preprocess_data as preprocess_module
from credit_default_prediction.preprocess_data import (
    handle_features_types,
    handle_missing_values,
    handle_outliers,
    log_transform_large_features,
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


def test_log_transformation_for_income_feature():
    """Given a dataframe containing loan applications data,
    When applying log transformation to it,
    Then the income feature is log scaled."""

    loan_data = pd.DataFrame(
        {
            "person_income": [59000, 9600, 80000, 6000000],
            "person_age": [22, 35, 50, 27],
        }
    )

    actual_clean_loan_data = log_transform_large_features(loan_data)

    expected_clean_loan_data = pd.DataFrame(
        {
            "person_income": np.log(loan_data["person_income"] + 1),
            "person_age": [22, 35, 50, 27],
        }
    )
    pd.testing.assert_frame_equal(
        expected_clean_loan_data,
        actual_clean_loan_data,
    )


def test_preprocess_pipeline_executes_steps_in_the_right_order(
    mocker: MockerFixture,
):
    """Given a dataframe containing loan applications data,
    When applying data preprocessing function to it,
    Then the function should:
        1. Handle missing values
        2. Handle outliers
        3. Handle features types
        4. Apply log transformation to large features
    In that order."""

    # Arrange
    sample_loan_data = pd.DataFrame(
        {
            "loan_int_rate": [11.84, np.nan, 12.5, 7.14, np.nan],
            "person_emp_length": [3, 0, 70, 60, 120],
            "person_income": [83000, 95000, 4000, 10000, 120000],
            "cb_person_default_on_file": ["Y", "N", "N", "Y", "N"],
        }
    )
    data_after_missing_values = handle_missing_values(sample_loan_data)
    data_after_outlier_treatment = handle_outliers(data_after_missing_values)
    data_after_default_on_file_as_boolean = handle_features_types(
        data_after_outlier_treatment
    )
    data_after_log_transformation = log_transform_large_features(
        data_after_default_on_file_as_boolean
    )
    # Spy test doubles
    spy_handle_missing_values = mocker.spy(
        preprocess_module,
        "handle_missing_values",
    )
    spy_handle_outliers = mocker.spy(
        preprocess_module,
        "handle_outliers",
    )
    spy_cb_default_type_change = mocker.spy(
        preprocess_module,
        "handle_features_types",
    )
    spy_log_transformation = mocker.spy(
        preprocess_module,
        "log_transform_large_features",
    )

    # Act
    clean_loan_data = preprocess_data(sample_loan_data)

    # Assert
    pd.testing.assert_frame_equal(
        spy_handle_missing_values.spy_return,
        data_after_missing_values,
    )
    pd.testing.assert_frame_equal(
        spy_handle_outliers.spy_return, data_after_outlier_treatment
    )
    pd.testing.assert_frame_equal(
        spy_cb_default_type_change.spy_return,
        data_after_default_on_file_as_boolean,
    )
    pd.testing.assert_frame_equal(
        spy_log_transformation.spy_return,
        data_after_log_transformation,
    )
    pd.testing.assert_frame_equal(
        clean_loan_data,
        data_after_log_transformation,
    )
