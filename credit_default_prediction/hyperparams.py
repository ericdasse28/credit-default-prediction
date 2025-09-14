from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class HyperParams:
    learning_rate: float
    max_depth: float = 4
    min_child_weight: float = 1

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, hyper_params: dict) -> HyperParams:
        return cls(**hyper_params)
