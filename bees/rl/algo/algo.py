from abc import ABC, abstractmethod
from typing import Tuple

from torch import optim

from bees.rl.model import Policy
from bees.rl.storage import RolloutStorage


class Algo(ABC):
    def __init__(self) -> None:
        self.actor_critic: Policy
        self.optimizer: optim.Optimizer
        self.lr: float

    @abstractmethod
    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        pass
