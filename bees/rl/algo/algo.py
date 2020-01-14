""" Abstract base class for RL algorithm implementations. """
from abc import ABC, abstractmethod
from typing import Tuple

from torch import optim

from bees.rl.model import Policy
from bees.rl.storage import RolloutStorage

# pylint: disable=too-few-public-methods


class Algo(ABC):
    """ Abstract base class for RL algorithm implementations. """

    def __init__(self) -> None:
        self.actor_critic: Policy
        self.optimizer: optim.Optimizer
        self.lr: float

    @abstractmethod
    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        """
        Execute a weight update given the experience from ``rollouts``.

        Parameters
        ----------
        rollouts : ``RolloutStorage``.
            Container with ``torch.Tensor``s of exprience to replay.

        Returns
        -------
        value_loss_epoch : ``float``.
            The value loss for current epoch.
        action_loss_epoch : ``float``.
            The action loss for current epoch.
        dist_entropy_epoch : ``float``.
            The distribution entropy for current epoch.
        """
        raise NotImplementedError
