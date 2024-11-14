"""Data preprocessing script."""

import argparse

import pandas as pd

NORMAL_MAX_EMP_LENGTH = 60


def handle_missing_values(loan_data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with the credit applications data."""

    # Drop rows with missing loan interest rates
    clean_loan_data = loan_data.dropna(subset=["loan_int_rate"])

    # Impute missing employment lengths
    clean_loan_data = clean_loan_data.fillna(
        {"person_emp_length": clean_loan_data["person_emp_length"].median()},
    )

    return clean_loan_data


def _get_emp_length_outlier_indices(loan_data: pd.DataFrame) -> pd.Index:
    """Returns indices of rows that contain outlier
    employment lengths.

    Here, an employment length is considered an outlier
    if it exceeds 60 years old."""

    EMP_LENGTH_THRESHOLD = 60

    outlier_filter = loan_data["person_emp_length"] > EMP_LENGTH_THRESHOLD
    indices = loan_data[outlier_filter].index

    return indices


def handle_outliers(loan_data: pd.DataFrame) -> pd.DataFrame:
    """Handle outlier values within the loan applications data."""

    # Drop rows with outlier employment lengths
    indices = _get_emp_length_outlier_indices(loan_data)
    clean_loan_data = loan_data.drop(index=indices)

    return clean_loan_data


def remove_outliers(loan_data: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers from `loan_data` in place."""

    outlier_filter = loan_data["person_emp_length"] > NORMAL_MAX_EMP_LENGTH
    indices = loan_data[outlier_filter].index

    return loan_data.drop(indices)


def remove_missing_loan_interests_rows(
    loan_data: pd.DataFrame,
) -> pd.DataFrame:

    return loan_data.dropna(subset=["loan_int_rate"])


def replace_missing_emp_length(loan_data: pd.DataFrame) -> pd.DataFrame:
    loan_data["person_emp_length"] = loan_data["person_emp_length"].fillna(
        loan_data["person_emp_length"].median()
    )
    return loan_data


def remove_unnecessary_rows(loan_data):
    clean_loan_data = remove_outliers(loan_data)
    clean_loan_data = remove_missing_loan_interests_rows(clean_loan_data)
    return clean_loan_data


def onehot_encode_str_columns(loan_data: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(loan_data)


def select_important_columns(
    loan_data: pd.DataFrame,
    important_columns: list[str],
):
    return loan_data[important_columns]


def preprocess(
    loan_data: pd.DataFrame,
) -> pd.DataFrame:
    clean_loan_data = remove_unnecessary_rows(loan_data)
    clean_loan_data = replace_missing_emp_length(clean_loan_data)

    return clean_loan_data


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path")
    parser.add_argument("--preprocessed-data-path")

    return parser.parse_args()


def main():
    args = _get_arguments()

    loan_data = pd.read_csv(args.raw_data_path)
    loan_data = preprocess(loan_data)
    loan_data.to_csv(args.preprocessed_data_path, index=False)
