""" Agent object for instantiating agents in the environment. """

from typing import Tuple
from policy import Policy


class Agent:
    """ An agent with position and health attributes. """

    def __init__(self, pos: Tuple[int] = None, health: float = 1) -> None:
        """ ``health`` ranges in ``[0, 1]``. """
        self.pos = pos
        self.health = health

        self.policy = Policy()
        self.observation = None

    def get_action(self):
        """ Uses the policy to choose an action based on the observation. """
        return self.policy.get_action(self.observation, self.health)

    def reset(self):
        """ Reset the agent. """
        pass
