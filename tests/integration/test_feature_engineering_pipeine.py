import pandas as pd
from pandas.testing import assert_frame_equal

from credit_default_prediction.feature_engineering import FeatureEngineer


def test_engineer_features():
    """To enhance the predictive power of the dataset,
    Given a dataframe that contains clean loan applications data,
    We need to perform feature engineering on it."""

    clean_loan_data = pd.DataFrame(
        {
            "person_age": [21, 25, 23, 24],
            "person_income": [
                33600,
                42000,
                49500,
                75000,
            ],
            "person_home_ownership": ["OWN", "MORTGAGE", "RENT", "RENT"],
            "person_emp_length": [5.0, 1.0, 4.0, 8.0],
            "loan_intent": ["EDUCATION", "MEDICAL", "MEDICAL", "MEDICAL"],
            "loan_grade": ["B", "C", "C", "C"],
            "loan_amnt": [
                5000,
                6000,
                6000,
                9600,
            ],
            "loan_int_rate": [11.14, 12.87, 15.23, 14.27],
            "loan_status": [0, 1, 1, 1],
            "loan_percent_income": [0.1, 0.57, 0.53, 0.55],
            "cb_person_default_on_file": [0, 0, 0, 1],
            "cb_person_cred_hist_length": [2, 3, 2, 4],
        }
    )
    feature_engineer = FeatureEngineer()

    actual_loan_data = feature_engineer.fit_transform(clean_loan_data)

    expected_loan_data = pd.DataFrame(
        {
            "person_income": [
                10.422311107413181,
                10.645448706505872,
                10.809748150372926,
                11.225256725762893,
            ],
            "person_emp_length": [5.0, 1.0, 4.0, 8.0],
            "person_age": [21, 25, 23, 24],
            "loan_percent_income": [0.1, 0.57, 0.53, 0.55],
            "loan_int_rate": [11.14, 12.87, 15.23, 14.27],
            "loan_amnt": [
                8.517393171418904,
                8.699681400989514,
                8.699681400989514,
                9.169622538697624,
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
