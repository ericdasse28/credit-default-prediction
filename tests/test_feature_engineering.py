import pandas as pd
from pandas.testing import assert_frame_equal

from credit_default_prediction.feature_engineering import engineer_features


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
