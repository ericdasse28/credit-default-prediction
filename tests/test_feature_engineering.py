import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from credit_default_prediction.feature_engineering import (
    engineer_features,
    log_transform_large_features,
)


def test_engineer_features():
    """To enhance the predictive power of the dataset,
    Given a dataframe that contains clean loan applications data,
    We need to perform feature engineering on it."""

    clean_loan_data = pd.DataFrame(
        {
            "person_age": [21, 25, 23, 24],
            "person_income": [
                9.169622538697624,
                9.169622538697624,
                11.089820688682373,
                10.904137815028022,
            ],
            "person_home_ownership": ["OWN", "MORTGAGE", "RENT", "RENT"],
            "person_emp_length": [5.0, 1.0, 4.0, 8.0],
            "loan_intent": ["EDUCATION", "MEDICAL", "MEDICAL", "MEDICAL"],
            "loan_grade": ["B", "C", "C", "C"],
            "loan_amnt": [
                6.90875477931522,
                8.612685172875459,
                10.463131911491967,
                10.463131911491967,
            ],
            "loan_int_rate": [11.14, 12.87, 15.23, 14.27],
            "loan_status": [0, 1, 1, 1],
            "loan_percent_income": [0.1, 0.57, 0.53, 0.55],
            "cb_person_default_on_file": [0, 0, 0, 1],
            "cb_person_cred_hist_length": [2, 3, 2, 4],
        }
    )

    actual_loan_data = engineer_features(clean_loan_data)

    expected_loan_data = pd.DataFrame(
        {
            "person_income": [
                9.169622538697624,
                9.169622538697624,
                11.089820688682373,
                10.904137815028022,
            ],
            "person_emp_length": [5.0, 1.0, 4.0, 8.0],
            "person_age": [21, 25, 23, 24],
            "loan_percent_income": [0.1, 0.57, 0.53, 0.55],
            "loan_int_rate": [11.14, 12.87, 15.23, 14.27],
            "loan_amnt": [
                6.90875477931522,
                8.612685172875459,
                10.463131911491967,
                10.463131911491967,
            ],
            "loan_status": [0, 1, 1, 1],
            "loan_grade_B": [True, False, False, False],
            "loan_grade_C": [False, True, True, True],
            "loan_intent_EDUCATION": [True, False, False, False],
            "loan_intent_MEDICAL": [False, True, True, True],
            "person_home_ownership_MORTGAGE": [False, True, False, False],
            "person_home_ownership_OWN": [True, False, False, False],
            "person_home_ownership_RENT": [False, False, True, True],
        }
    )
    assert_frame_equal(expected_loan_data, actual_loan_data)


def test_log_transformation_for_large_features():
    """Given a dataframe containing loan applications data,
    When applying log transformation to it,
    Then the loan amount and income features are log scaled."""

    loan_data = pd.DataFrame(
        {
            "loan_amnt": [19000.0, 5000.0, 1500.0, 10000.0],
            "person_income": [59000.0, 9600.0, 80000.0, 6000000.0],
            "person_age": [22, 35, 50, 27],
        }
    )

    actual_clean_loan_data = log_transform_large_features(loan_data)

    expected_clean_loan_data = pd.DataFrame(
        {
            "loan_amnt": np.log(loan_data["loan_amnt"] + 1),
            "person_income": np.log(loan_data["person_income"] + 1),
            "person_age": [22, 35, 50, 27],
        }
    )
    pd.testing.assert_frame_equal(
        expected_clean_loan_data,
        actual_clean_loan_data,
    )
