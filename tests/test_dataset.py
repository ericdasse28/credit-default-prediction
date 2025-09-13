import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from credit_default_prediction.dataset import Dataset, collect_loan_dataset_from_path


@pytest.fixture
def dummy_dataset_path(tmp_path):
    dataset_csv_path = tmp_path / "data.csv"

    dummy_data = pd.DataFrame(
        {
            "person_age": [22, 25, 40],
            "person_income": [35000, 45000, 52000],
            "loan_intent": ["PERSONAL", "MEDICAL", "VENTURE"],
            "loan_status": [0, 1, 1],
        }
    )
    dummy_data.to_csv(dataset_csv_path, index=False)

    return dataset_csv_path


def test_collect_loan_dataset():
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

    actual_dataset = Dataset.from_dataframe(dummy_data)

    expected_dataset = Dataset(X=X, y=y)
    assert_frame_equal(expected_dataset.X, actual_dataset.X)
    assert_series_equal(expected_dataset.y, actual_dataset.y)


def test_collect_loan_dataset_from_path(dummy_dataset_path):
    actual_dataset = collect_loan_dataset_from_path(dummy_dataset_path)

    X = pd.DataFrame(
        {
            "person_age": [22, 25, 40],
            "person_income": [35000, 45000, 52000],
            "loan_intent": ["PERSONAL", "MEDICAL", "VENTURE"],
        }
    )
    y = pd.Series([0, 1, 1], name="loan_status")
    expected_dataset = Dataset(X=X, y=y)
    assert_frame_equal(expected_dataset.X, actual_dataset.X)
    assert_series_equal(expected_dataset.y, expected_dataset.y)
