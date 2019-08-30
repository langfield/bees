""" A dummy policy for bees agents. """
import random

from constants import *

# pylint: disable=too-few-public-methods
class Policy:
    """ Policy class defining random actions. """

    @staticmethod
    def get_action(_obs, _agent_health):
        """ Returns a random action. """
        move = random.choice([LEFT, RIGHT, UP, DOWN, STAY])
        consume = random.choice([EAT, NO_EAT])

        return (move, consume)
