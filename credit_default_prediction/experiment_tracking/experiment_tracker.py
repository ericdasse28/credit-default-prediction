from typing import Protocol

from credit_default_prediction.training import HyperParams


class ExperimentTracker(Protocol):
    def log_params(self, hyperparameters: HyperParams): ...
    def log_metrics(self, **metrics): ...
