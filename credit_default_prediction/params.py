from pathlib import Path

import yaml

from credit_default_prediction.split import SplitParams
from credit_default_prediction.training import HyperParams

PARAMS_FILE_PATH = Path(__file__).parent.parent / "params.yaml"


def _load_pipeline_params() -> dict:
    with open(PARAMS_FILE_PATH) as params_file:
        pipeline_params = yaml.safe_load(params_file)

    return pipeline_params


def _load_stage_params(stage_name: str) -> dict:
    pipeline_params = _load_pipeline_params()
    return pipeline_params[stage_name]


def _get_hyperparameters_from_config(params_file_path):
    with open(params_file_path) as params_file:
        params = yaml.safe_load(params_file)

    return params["train"]


def get_hyperparameters() -> HyperParams:
    hyper_params_dict = _get_hyperparameters_from_config(PARAMS_FILE_PATH)

    return HyperParams.from_dict(hyper_params_dict)


def get_important_features() -> list[str]:
    preprocess_params = _load_stage_params("feature_engineering")
    return preprocess_params["important_columns"]


def load_split_params() -> SplitParams:
    pipeline_params = _load_stage_params("split")
    return SplitParams(
        test_size=pipeline_params["test_size"],
        random_state=pipeline_params["random_state"],
    )
