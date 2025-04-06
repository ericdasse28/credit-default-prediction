import numpy as np
import pandas as pd
from pytest_mock import MockerFixture

from credit_default_prediction import data_preprocessing
from credit_default_prediction.data_preprocessing import (
    handle_features_types,
    handle_missing_values,
    handle_outliers,
    preprocess_data,
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
            "loan_amnt": [13000.0, 2300.5, 1400.89, 120000.0, 13000.9],
            "cb_person_default_on_file": ["Y", "N", "N", "Y", "N"],
        }
    )
    data_after_missing_values = handle_missing_values(sample_loan_data)
    data_after_outlier_treatment = handle_outliers(data_after_missing_values)
    data_after_default_on_file_as_boolean = handle_features_types(
        data_after_outlier_treatment
    )
    # Spy test doubles
    spy_handle_missing_values = mocker.spy(
        data_preprocessing,
        "handle_missing_values",
    )
    spy_handle_outliers = mocker.spy(
        data_preprocessing,
        "handle_outliers",
    )
    spy_cb_default_type_change = mocker.spy(
        data_preprocessing,
        "handle_features_types",
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
        clean_loan_data,
        data_after_default_on_file_as_boolean,
    )
