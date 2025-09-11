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
            "person_age": [20, 50, 18, 65, 19],
            "person_emp_length": [2, 15, 1.3, 20, 3],
        }
    )
    preprocessor = build_preprocessing_pipeline()

    transformed = preprocessor.fit_transform(loan_applications)

    expected_clean_loan_data = np.array(
        [
            [11.84, 20, 2],
            [12.5, 18, 1.3],
            [7.14, 65, 20],
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

