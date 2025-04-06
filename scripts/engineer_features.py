import argparse

import pandas as pd

from credit_default_prediction.feature_engineering import engineer_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-data-path")
    parser.add_argument("--feature-store-path")
    args = parser.parse_args()

    preprocessed_data = pd.read_csv(args.preprocessed_data_path)
    preprocessed_data = engineer_features(preprocessed_data)
    # Save feature engineered features
    preprocessed_data.to_csv(args.feature_store_path, index=False)
