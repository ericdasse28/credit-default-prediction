import argparse
from pathlib import Path

from credit_default_prediction.split import SplitParams, split_data_from_path
from credit_default_prediction.tools.params import load_stage_params


def _get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path")
    parser.add_argument("--train-path")
    parser.add_argument("--test-path")

    return parser.parse_args()


def _load_split_params() -> SplitParams:
    pipeline_params = load_stage_params("split")
    return SplitParams(
        test_size=pipeline_params["test_size"],
        random_state=pipeline_params["random_state"],
    )


def main():
    args = _get_arguments()

    split_params = _load_split_params()
    split_data_dir = Path(args.train_path).parent
    split_data_from_path(
        raw_data_path=args.raw_data_path,
        split_data_dir=split_data_dir,
        split_params=split_params,
    )
