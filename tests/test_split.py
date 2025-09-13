import pandas as pd
import pytest

from credit_default_prediction.split import (
    SplitParams,
    split_data,
    split_data_from_path,
)


def test_split_data():
    """Given loan applications dataframe and split parameters,
    When we split the loan applications dataframe using said parameters,
    Then we obtain training and test data."""

    dummy_data = pd.DataFrame(
        {
            "person_age": [22, 25, 40],
            "person_income": [35000, 45000, 52000],
            "loan_intent": ["PERSONAL", "MEDICAL", "VENTURE"],
            "loan_status": [0, 1, 1],
        }
    )
    dummy_parameters = SplitParams(test_size=0.2, random_state=3)

    training_data, test_data = split_data(dummy_data, dummy_parameters)

    assert len(training_data.X) == 2
    assert len(training_data.y) == 2
    assert len(test_data.X) == 1
    assert len(test_data.y) == 1


def test_split_data_missing_loan_status():
    """Given loan applications dataframe with a missing 'loan_status' column
      and split parameters,
    When we split the loan applications dataframe using said parameters,
    Then a KeyError exception about the missing 'loan_status' column is raised."""

    dummy_data = pd.DataFrame(
        {
            "person_age": [22, 25, 40],
            "person_income": [35000, 45000, 52000],
            "loan_intent": ["PERSONAL", "MEDICAL", "VENTURE"],
        }
    )
    dummy_parameters = SplitParams(test_size=0.2, random_state=3)

    with pytest.raises(KeyError) as e_info:
        split_data(dummy_data, dummy_parameters)

    assert "loan_status" in str(e_info.value)


def test_split_data_from_path(tmp_path):
    """Given
    - a path to raw loan applications,
    - split parameters
    - a path to save split data
    When we split the loan applications,
    Then a training dataset and a test dataset
    are saved at the split data path."""

    raw_data_dir = tmp_path / "raw"
    raw_data_dir.mkdir()
    raw_data_path = raw_data_dir / "original.csv"
    split_data_dir = tmp_path / "split"
    split_data_dir.mkdir()
    # Pre-save raw data at intended path
    loan_applications = pd.DataFrame(
        {
            "person_age": [22, 25, 40],
            "person_income": [35000, 45000, 52000],
            "loan_intent": ["PERSONAL", "MEDICAL", "VENTURE"],
            "loan_status": [0, 1, 1],
        }
    )
    loan_applications.to_csv(raw_data_path, index=False)

    split_data_from_path(
        raw_data_path=raw_data_path,
        split_data_dir=split_data_dir,
        split_params=SplitParams(test_size=0.2, random_state=3),
    )

    train_file_path = split_data_dir / "train.csv"
    test_file_path = split_data_dir / "test.csv"
    assert train_file_path.exists()
    assert test_file_path.exists()
