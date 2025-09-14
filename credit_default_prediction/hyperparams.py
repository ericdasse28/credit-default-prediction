from __future__ import annotations

from dataclasses import asdict, dataclass

from credit_default_prediction import params


@dataclass
class HyperParams:
    learning_rate: float
    max_depth: float = 4
    min_child_weight: float = 1

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, hyper_params: dict[str, float]) -> HyperParams:
        return cls(**hyper_params)

    @classmethod
    def from_config(cls) -> HyperParams:
        hyperparams_config = params.get_hyperparameters()
        return cls.from_dict(hyperparams_config)
