"""Dataset-related functions."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class LoanApplications:
    X: pd.DataFrame
    y: pd.Series

    @property
    def data(self) -> pd.DataFrame:
        return pd.concat([self.X, self.y], axis=1)

    def save_to_csv(self, csv_path: os.PathLike):
        as_dataframe = pd.concat([self.X, self.y], axis=1)
        as_dataframe.to_csv(csv_path, index=False)

    @classmethod
    def from_dataframe(cls, loan_data: pd.DataFrame) -> LoanApplications:
        X = loan_data.drop("loan_status", axis=1)
        y = loan_data["loan_status"]

        return cls(X=X, y=y)

    @classmethod
    def from_path(
        cls, dataset_path: os.PathLike, columns: list[str] = None
    ) -> LoanApplications:
        loan_data = pd.read_csv(dataset_path)
        if columns:
            loan_data = loan_data[columns]

        loan_dataset = LoanApplications.from_dataframe(loan_data)

        return cls(X=loan_dataset.X, y=loan_dataset.y)
