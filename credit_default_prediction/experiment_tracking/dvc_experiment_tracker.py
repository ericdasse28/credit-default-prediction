from dvclive.live import Live

from credit_default_prediction.experiment_tracking.experiment_tracker import (
    ExperimentTracker,
)
from credit_default_prediction.training import HyperParams


class DVCExperimentTracker(ExperimentTracker):
    def __init__(self, live: Live) -> None:
        self._live = live

    def log_params(self, hyperparameters: HyperParams):
        self._live.log_params(hyperparameters.to_dict())  # type: ignore

    def log_metrics(self, **metrics):
        for metric, value in metrics.items():
            self._live.log_metric(metric, value)
