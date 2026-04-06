from typing import Protocol

from credit_default_prediction.hyper_params import HyperParams


class ExperimentTracker(Protocol):
    def log_params(self, hyperparameters: HyperParams): ...
    def log_metrics(self, **metrics): ...
