import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

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

    expected_clean_loan_data = pd.DataFrame(
        {
            "loan_int_rate": [11.84, 12.5, 7.14],
            "person_age": [20, 18, 65],
            "person_emp_length": [2, 1.3, 20],
        },
        index=[0, 2, 3],
    )
    assert_frame_equal(
        expected_clean_loan_data,
        transformed,
    )
