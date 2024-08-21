"""Data preprocessing script."""

import argparse

import pandas as pd

NORMAL_MAX_EMP_LENGTH = 60


def remove_outliers(loan_data: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers from `loan_data` in place."""

    outlier_filter = loan_data["person_emp_length"] > NORMAL_MAX_EMP_LENGTH
    indices = loan_data[outlier_filter].index

    return loan_data.drop(indices)


def remove_missing_loan_interests_rows(
    loan_data: pd.DataFrame,
) -> pd.DataFrame:
    missing_loan_int_filter = loan_data["loan_int_rate"].isnull()
    indices = loan_data[missing_loan_int_filter].index

    return loan_data.drop(indices)


def replace_missing_emp_length(loan_data: pd.DataFrame) -> pd.DataFrame:
    loan_data["person_emp_length"] = loan_data["person_emp_length"].fillna(
        loan_data["person_emp_length"].median()
    )
    return loan_data


def remove_unnecessary_rows(loan_data):
    clean_loan_data = remove_outliers(loan_data)
    clean_loan_data = remove_missing_loan_interests_rows(clean_loan_data)
    return clean_loan_data


def preprocess(loan_data: pd.DataFrame) -> pd.DataFrame:
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
    loan_data.to_csv(args.preprocessed_data_path)
