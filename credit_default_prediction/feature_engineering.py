import argparse

import pandas as pd

from credit_default_prediction import params


def get_important_features():
    preprocess_params = params.load_stage_params("feature_engineering")
    return preprocess_params["important_columns"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-data-path")
    parser.add_argument("--feature-store-path")
    args = parser.parse_args()

    preprocessed_data = pd.read_csv(args.preprocessed_data_path)
    preprocessed_data = preprocessed_data[get_important_features()]
    preprocessed_data = pd.get_dummies(preprocessed_data)
    preprocessed_data.to_csv(args.feature_store_path, index=False)
