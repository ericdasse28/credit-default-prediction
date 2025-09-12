import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from credit_default_prediction.preprocessing import build_preprocessing_pipeline


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
    preprocessor = build_preprocessing_pipeline()

    transformed = preprocessor.fit_transform(loan_applications)

    expected_clean_loan_data = np.array(
        [
            [2, 11.84, 20],
            [1.3, 12.5, 18],
            [20, 7.14, 25],
        ]
    )
    assert_array_equal(transformed, expected_clean_loan_data)


def test_missing_person_emp_length_are_imputed():
    """Given loan applications,
    When we preprocess them,
    Then rows with missing employment lengths are imputed.
    """

    loan_applications = pd.DataFrame(
        {
            "person_age": [40, 35, 50, 40, 19],
            "loan_int_rate": [11.5, 8.3, 4.5, 6.9, 7.8],
            "person_emp_length": [np.nan, 12, 13, 25, 3],
        }
    )
    preprocessor = build_preprocessing_pipeline()

    transformed = preprocessor.fit_transform(loan_applications)

    median_emp_length = loan_applications["person_emp_length"].median()
    expected_clean_loan_data = np.array(
        [
            [median_emp_length, 40, 11.5],
            [12, 35, 8.3],
            [13, 50, 4.5],
            [25, 40, 6.9],
            [3, 19, 7.8],
        ]
    )
    assert_array_equal(
        transformed,
        expected_clean_loan_data,
    )


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
    preprocessor = build_preprocessing_pipeline()

    transformed = preprocessor.fit_transform(loan_applications)

    expected_clean_loan_applications = np.array(
        [
            [3, 11.5, 1],
            [0, 8.3, 1],
            [60, 6.9, 0],
        ]
    )
    assert_array_equal(
        transformed,
        expected_clean_loan_applications,
    )


def test_categorical_features_are_one_hot_encoded():
    """Given loan applications and a list of categorical features
    of interest within it,
    When we preprocess them,
    Then the provided categorical features are one-hot encoded.
    """

    loan_applications = pd.DataFrame(
        {
            "person_emp_length": [3, 0, 70, 60, 120],
            "loan_int_rate": [11.5, 8.3, 4.5, 6.9, 7.8],
            "loan_grade": ["B", "C", "C", "C", "B"],
            "loan_intent": ["EDUCATION", "MEDICAL", "MEDICAL", "MEDICAL", "MEDICAL"],
            "loan_status": [1, 1, 0, 0, 1],
        }
    )
    preprocessor = build_preprocessing_pipeline(
        categorical_features=["loan_grade", "loan_intent"]
    )

    transformed = preprocessor.fit_transform(loan_applications)

    expected_clean_loan_applications = np.array(
        [
            [True, False, True, False, 3, 11.5, 1],
            [False, True, False, True, 0, 8.3, 1],
            [False, True, False, True, 60, 6.9, 0],
        ]
    )
    assert_array_equal(
        transformed,
        expected_clean_loan_applications,
    )
