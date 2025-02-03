import typing

from dvclive import Live


def save_model_metrics(
    metrics: dict, phase: typing.Literal["cross_validation", "test"] = "test"
):
    with Live(resume=True) as live:
        for metric in metrics.keys():
            live.log_metric(f"{phase}/{metric}", metrics[metric])
