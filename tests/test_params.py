import pytest
import yaml

from credit_default_prediction.tools import params


@pytest.mark.parametrize(
    "params_file_content",
    [
        {
            "split": {"test_size": 0.2, "random_state": 123},
            "train": {"learning_rate": 0.5, "max_depth": 3},
        },
        {"train": {"min_child_weight": 0.3}},
    ],
)
def test_get_hyperparameters_from_config(tmp_path, params_file_content):
    fake_params_file_path = tmp_path / "fake_params.yaml"
    with open(fake_params_file_path, "w") as fake_params_file:
        yaml.dump(params_file_content, fake_params_file)

    actual_hyperparameters = params._get_hyperparameters_from_config(
        fake_params_file_path,
    )

    expected_hyperparameters = params_file_content["train"]
    assert expected_hyperparameters == actual_hyperparameters


def test_get_hyperparameters(mocker):
    fake_expected_hyperparameters = {"learning_rate": 0.3}
    mocked_get_hyperparameters_from_config = mocker.patch(
        "credit_default_prediction.tools.params._get_hyperparameters_from_config",  # noqa
        return_value=fake_expected_hyperparameters,
    )

    actual_hyperparameters = params.get_hyperparameters()

    mocked_get_hyperparameters_from_config.assert_called_once_with(
        params.PARAMS_FILE_PATH,
    )
    assert fake_expected_hyperparameters == actual_hyperparameters


def test_params_file_presence():
    assert params.PARAMS_FILE_PATH.exists()
