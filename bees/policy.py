""" A dummy policy for bees agents. """
import random
from typing import Dict, Any, Tuple

import numpy as np

# pylint: disable=too-few-public-methods, bad-continuation
class Policy:
    """
    Policy class defining random actions.

    Parameters
    ----------
    consts : ``Dict[str, Any]``.
        Dictionary of various constants.
    """

    def __init__(self, consts: Dict[str, Any]) -> None:

        # pylint: disable=invalid-name
        self.consts = consts
        self.LEFT = consts["LEFT"]
        self.RIGHT = consts["RIGHT"]
        self.UP = consts["UP"]
        self.DOWN = consts["DOWN"]
        self.STAY = consts["STAY"]
        self.EAT = consts["EAT"]
        self.NO_EAT = consts["NO_EAT"]
        self.MATE = consts["MATE"]
        self.NO_MATE = consts["NO_MATE"]

    def get_action(
        self, _obs: Tuple[Tuple[Tuple[int, ...], ...], ...], _agent_health: float
    ) -> np.ndarray:
        """
        Returns a random action.

        Parameters
        ----------
        _obs : ``Tuple[Tuple[Tuple[int, ...], ...], ...]``.
            The agent observation.
        _agent_health : ``float``.
            Current agent health.

        Returns
        -------
        action: ``np.ndarray``.
            5D Multi-Binary numpy array representing action.
        """
        move = random.choice([self.LEFT, self.RIGHT, self.UP, self.DOWN, self.STAY])
        consume = random.choice([self.EAT, self.NO_EAT])
        mate = random.choice([self.MATE, self.NO_MATE])

        # Convert to 5-D Multi-Binary numpy array
        # HARDCODE
        action = np.zeros(5)
        if move == self.LEFT:
            action[0] = 1
            action[1] = 0
            action[2] = 0
        elif move == self.RIGHT:
            action[0] = 1
            action[1] = 0
            action[2] = 1
        elif move == self.DOWN:
            action[0] = 1
            action[1] = 1
            action[2] = 0
        elif move == self.UP:
            action[0] = 1
            action[1] = 1
            action[2] = 1
        elif move == self.STAY:
            action[0] = 0
            action[1] = 0
            action[2] = 0
        action[3] = 0 if consume == self.EAT else 1
        action[4] = 0 if mate == self.MATE else 1

        return action
