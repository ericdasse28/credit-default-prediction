from pathlib import Path

import yaml

PARAMS_FILE_PATH = Path(__file__).parent.parent / "params.yaml"


def _load_pipeline_params() -> dict:
    with open(PARAMS_FILE_PATH) as params_file:
        pipeline_params = yaml.safe_load(params_file)

    return pipeline_params


def load_stage_params(stage_name: str) -> dict:
    pipeline_params = _load_pipeline_params()
    return pipeline_params[stage_name]


def get_hyperparameters() -> dict[str, float]:
    with open(PARAMS_FILE_PATH) as params_file:
        params = yaml.safe_load(params_file)

    return params["train"]


def get_important_features() -> list[str]:
    preprocess_params = load_stage_params("feature_engineering")
    return preprocess_params["important_columns"]
