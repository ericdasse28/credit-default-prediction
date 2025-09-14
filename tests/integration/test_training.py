import pandas as pd
from sklearn.base import BaseEstimator

from credit_default_prediction.hyperparams import HyperParams
from credit_default_prediction.training import train


def test_train():
    X = pd.DataFrame(
        {
            "person_income": [
                10.422311107413181,
                10.645448706505872,
                10.809748150372926,
                11.225256725762893,
            ],
            "person_emp_length": [5.0, 1.0, 4.0, 8.0],
            "person_age": [21, 25, 23, 24],
            "loan_percent_income": [0.1, 0.57, 0.53, 0.55],
            "loan_int_rate": [11.14, 12.87, 15.23, 14.27],
            "loan_amnt": [
                8.517393171418904,
                8.699681400989514,
                8.699681400989514,
                9.169622538697624,
            ],
            "loan_grade": ["B", "C", "C", "C"],
            "loan_intent": ["EDUCATION", "MEDICAL", "MEDICAL", "MEDICAL"],
            "person_home_ownership": ["OWN", "MORTGAGE", "RENT", "RENT"],
            "loan_status": [1, 0, 1, 1],
        }
    )
    y = pd.Series([0, 1, 1, 1], name="loan_status")
    hyperparameters = HyperParams(learning_rate=0.2, max_depth=5, min_child_weight=7)

    trained_model = train(X, y, hyperparameters)

    assert isinstance(trained_model, BaseEstimator)
