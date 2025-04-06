import pandas as pd

from credit_default_prediction.training import HyperParams, train


class FakeClassifier:
    def __init__(self):
        self.has_trained = False

    def fit(self, X, y):
        self.has_trained = True


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
            "loan_grade_B": [True, False, False, False],
            "loan_grade_C": [False, True, True, True],
            "loan_intent_EDUCATION": [True, False, False, False],
            "loan_intent_MEDICAL": [False, True, True, True],
            "person_home_ownership_MORTGAGE": [False, True, False, False],
            "person_home_ownership_OWN": [True, False, False, False],
            "person_home_ownership_RENT": [False, False, True, True],
        }
    )
    y = pd.Series([0, 1, 1, 1], name="loan_status")
    hyper_parameters = HyperParams(learning_rate=0.2, max_depth=5, min_child_weight=7)

    trained_model = train(X, y, hyper_parameters)

    assert trained_model
