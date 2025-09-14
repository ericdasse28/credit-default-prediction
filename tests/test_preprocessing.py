import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from credit_default_prediction.preprocessing import (
    build_infered_transformers,
    rule_based_preprocessing,
)


def test_missing_loan_interests_are_dropped():
    """Given loan applications,
    When we preprocess them,
    Then rows with missing loan interests should be dropped.
    """

    loan_applications = pd.DataFrame(
        {
            "loan_int_rate": [11.84, np.nan, 12.5, 7.14, np.nan],
            "person_age": [20, 50, 18, 25, 19],
            "person_emp_length": [2, 15, 1.3, 20, 3],
        }
    )

    transformed = rule_based_preprocessing(loan_applications)

    expected_clean_loan_data = pd.DataFrame(
        {
            "loan_int_rate": [11.84, 12.5, 7.14],
            "person_age": [20, 18, 25],
            "person_emp_length": [2, 1.3, 20],
        },
        index=[0, 2, 3],
    )
    assert_frame_equal(transformed, expected_clean_loan_data)


def test_rows_with_outlier_employment_lengths_are_dropped():
    """Given loan applications,
    When we preprocess them,
    Then rows with employment lengths greater than 60 years are dropped.
    """

    loan_applications = pd.DataFrame(
        {
            "person_emp_length": [3, 0, 70, 60, 120],
            "loan_int_rate": [11.5, 8.3, 4.5, 6.9, 7.8],
            "loan_status": [1, 1, 0, 0, 1],
        }
    )

    transformed = rule_based_preprocessing(loan_applications)

    expected_clean_loan_applications = pd.DataFrame(
        {
            "person_emp_length": [3, 0, 60],
            "loan_int_rate": [11.5, 8.3, 6.9],
            "loan_status": [1, 1, 0],
        },
        index=[0, 1, 3],
    )
    assert_frame_equal(
        transformed,
        expected_clean_loan_applications,
    )


def test_missing_person_emp_length_are_imputed():
    """Given loan applications,
    When we preprocess them,
    Then rows with missing employment lengths are imputed.
    """

    loan_features = pd.DataFrame(
        {
            "person_age": [40, 35, 50, 40, 19],
            "loan_int_rate": [11.5, 8.3, 4.5, 6.9, 7.8],
            "person_emp_length": [np.nan, 12, 13, 25, 3],
        }
    )
    preprocessor = build_infered_transformers(
        numeric_features=loan_features.columns,
        categorical_features=[],
    )

    transformed = preprocessor.fit_transform(loan_features)

    median_emp_length = loan_features["person_emp_length"].median()
    expected_clean_loan_data = np.array(
        [
            [40, 11.5, median_emp_length],
            [35, 8.3, 12],
            [50, 4.5, 13],
            [40, 6.9, 25],
            [19, 7.8, 3],
        ]
    )
    assert_array_equal(
        transformed,
        expected_clean_loan_data,
    )


def test_categorical_features_are_one_hot_encoded():
    """Given loan applications and a list of categorical features
    of interest within it,
    When we preprocess them,
    Then the provided categorical features are one-hot encoded.
    """

    loan_features = pd.DataFrame(
        {
            "person_emp_length": [3, 0, 60],
            "loan_int_rate": [11.5, 8.3, 6.9],
            "loan_grade": ["B", "C", "C"],
            "loan_intent": ["EDUCATION", "MEDICAL", "MEDICAL"],
        }
    )
    preprocessor = build_infered_transformers(
        numeric_features=["person_emp_length", "loan_int_rate"],
        categorical_features=["loan_grade", "loan_intent"],
    )

    transformed = preprocessor.fit_transform(loan_features)

    expected_clean_loan_applications = np.array(
        [
            [3, 11.5, True, False, True, False],
            [0, 8.3, False, True, False, True],
            [60, 6.9, False, True, False, True],
        ]
    )
    assert_array_equal(
        transformed,
        expected_clean_loan_applications,
    )
