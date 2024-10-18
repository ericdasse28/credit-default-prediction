from pathlib import Path

import yaml

PARAMS_FILE_PATH = Path(__file__).parent.parent / "params.yaml"


def load_pipeline_params() -> dict:
    with open(PARAMS_FILE_PATH) as params_file:
        pipeline_params = yaml.safe_load(params_file)

    return pipeline_params


def load_stage_params(stage_name: str) -> dict:
    pipeline_params = load_pipeline_params()
    return pipeline_params[stage_name]
