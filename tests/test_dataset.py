import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from credit_default_prediction.dataset import Dataset, get_features_and_labels


def test_get_features_and_labels():
    dummy_data = pd.DataFrame(
        {
            "person_age": [22, 25, 40],
            "person_income": [35000, 45000, 52000],
            "loan_intent": ["PERSONAL", "MEDICAL", "VENTURE"],
            "loan_status": [0, 1, 1],
        }
    )
    X = pd.DataFrame(
        {
            "person_age": [22, 25, 40],
            "person_income": [35000, 45000, 52000],
            "loan_intent": ["PERSONAL", "MEDICAL", "VENTURE"],
        }
    )
    y = pd.Series([0, 1, 1], name="loan_status")

    actual_dataset = get_features_and_labels(dummy_data)

    expected_dataset = Dataset(X=X, y=y)
    assert_frame_equal(expected_dataset.X, actual_dataset.X)
    assert_series_equal(expected_dataset.y, actual_dataset.y)
