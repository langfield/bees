""" A dummy policy for bees agents. """
import random
from typing import Dict, Any, Tuple

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
    ) -> Tuple[int, int, int]:
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
        move : ``int``.
            Move subaction.
        consume : ``int``.
            Consume subaction.
        mate : ``int``.
            Mate subaction.
        """
        # DEBUG
        print("Using random policy.")

        move = random.choice([self.LEFT, self.RIGHT, self.UP, self.DOWN, self.STAY])
        consume = random.choice([self.EAT, self.NO_EAT])
        mate = random.choice([self.MATE, self.NO_MATE])

        return move, consume, mate
