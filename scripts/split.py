import argparse
from pathlib import Path

from credit_default_prediction import params
from credit_default_prediction.split import split_data_from_path


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path")
    parser.add_argument("--train-path")
    parser.add_argument("--test-path")

    return parser.parse_args()


def main():
    args = _get_arguments()
    split_data_dir = Path(args.train_path).parent
    split_params = params.load_split_params()

    split_data_from_path(
        raw_data_path=args.raw_data_path,
        split_data_dir=split_data_dir,
        split_params=split_params,
    )
