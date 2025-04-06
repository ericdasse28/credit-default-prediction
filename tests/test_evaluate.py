import pandas as pd
import pytest

from credit_default_prediction.evaluation import generate_feature_importance_data


@pytest.fixture
def model():
    class FakeBooster:
        def get_score(self):
            return {"feature_1": 196.0, "feature_2": 223.0}

    class FakeModel:
        def get_booster(self):
            return FakeBooster()

    return FakeModel()


def test_generate_feature_importance_data(model):
    actual_series = generate_feature_importance_data(model)

    expected_series = pd.DataFrame(
        {
            "feature_name": ["feature_1", "feature_2"],
            "feature_importance": [196.0, 223.0],
        }
    )
    pd.testing.assert_frame_equal(actual_series, expected_series)
