import argparse

import pandas as pd

from credit_default_prediction.data_preprocessing import preprocess_data


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path")
    parser.add_argument("--preprocessed-data-path")

    return parser.parse_args()


def main():
    args = _get_arguments()

    loan_data = pd.read_csv(args.raw_data_path)
    loan_data = preprocess_data(loan_data)
    loan_data.to_csv(args.preprocessed_data_path, index=False)
