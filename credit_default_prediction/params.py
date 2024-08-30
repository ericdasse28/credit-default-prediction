from pathlib import Path

import yaml

PARAMS_FILE_PATH = Path(__file__).parent.parent / "params.yaml"


def load_pipeline_params():
    with open(PARAMS_FILE_PATH) as params_file:
        pipeline_params = yaml.safe_load(params_file)

    return pipeline_params
