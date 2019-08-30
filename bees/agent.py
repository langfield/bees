""" Agent object for instantiating agents in the environment. """

from typing import Tuple
from policy import Policy


class Agent:
    """ An agent with position and health attributes. """

    def __init__(
        self, config: dict, pos: Tuple[int, int] = None, initial_health: float = 1
    ) -> None:
        """ ``health`` ranges in ``[0, 1]``. """
        self.pos = pos
        self.initial_health = initial_health
        self.health = self.initial_health

        self.policy = Policy(config)
        self.observation = None
        self.total_reward = 0.0

    def get_action(self):
        """ Uses the policy to choose an action based on the observation. """
        return self.policy.get_action(self.observation, self.health)

    def update_total_reward(self, new_reward):
        """ Updates total reward. """
        self.total_reward += new_reward

    def reset(self):
        """ Reset the agent. """
        self.health = self.initial_health
        self.total_reward = 0.0
        return self.observation

    def __repr__(self):
        output = "Health: %f, " % self.health
        output += "Total reward: %f\n" % self.total_reward
        return output
