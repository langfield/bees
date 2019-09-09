""" A dummy policy for bees agents. """
import random
from typing import Dict, Any

# pylint: disable=too-few-public-methods
class Policy:
    """ Policy class defining random actions. """

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

    def get_action(self, _obs, _agent_health):
        """ Returns a random action. """
        move = random.choice([self.LEFT, self.RIGHT, self.UP, self.DOWN, self.STAY])
        consume = random.choice([self.EAT, self.NO_EAT])
        mate = random.choice([self.MATE, self.NO_MATE])

        return (move, consume, mate)
