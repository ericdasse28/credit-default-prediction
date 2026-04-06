from dvclive.live import Live

from credit_default_prediction.experiment_tracking.experiment_tracker import (
    ExperimentTracker,
)


class DVCExperimentTracker(ExperimentTracker):
    def __init__(self, live: Live) -> None:
        self._live = live

    def log_params(self, **hyperparameters):
        self._live.log_params(**hyperparameters)

    def log_metrics(self, **metrics):
        for metric, value in metrics.items():
            self._live.log_metric(metric, value)
