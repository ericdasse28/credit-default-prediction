"""Data preprocessing script."""

import pandas as pd

NORMAL_MAX_EMP_LENGTH = 60


def remove_outliers(loan_data: pd.DataFrame):
    """Remove outliers from `loan_data` in place."""

    outlier_filter = loan_data["person_emp_length"] > NORMAL_MAX_EMP_LENGTH
    indices = loan_data[outlier_filter].index
    return loan_data.drop(indices)
